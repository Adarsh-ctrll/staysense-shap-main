const API_BASE_URL = ""; // same-origin when main_shap.py serves this frontend

const PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;
const EXPLAIN_ENDPOINT = `${API_BASE_URL}/explain`;
const REPORT_ENDPOINT = `${API_BASE_URL}/report`;
const HEALTH_ENDPOINT = `${API_BASE_URL}/`;

const examples = [
    {
        latitude: 40.7484,
        longitude: -73.9857,
        price: 120,
        minimum_nights: 2,
        number_of_reviews: 84,
        reviews_per_month: 2.3,
        calculated_host_listings_count: 1,
        availability_365: 210,
        neighbourhood_group: "Manhattan",
        neighbourhood: "Midtown"
    },
    {
        latitude: 40.6782,
        longitude: -73.9442,
        price: 55,
        minimum_nights: 1,
        number_of_reviews: 210,
        reviews_per_month: 4.1,
        calculated_host_listings_count: 3,
        availability_365: 300,
        neighbourhood_group: "Brooklyn",
        neighbourhood: "Bedford-Stuyvesant"
    },
    {
        latitude: 40.7282,
        longitude: -73.7949,
        price: 38,
        minimum_nights: 3,
        number_of_reviews: 12,
        reviews_per_month: 0.6,
        calculated_host_listings_count: 1,
        availability_365: 90,
        neighbourhood_group: "Queens",
        neighbourhood: "Flushing"
    }
];

let exIndex = 0;
let probChart = null;
let lastPayload = null;
let lastExplanation = null;

const $ = id => document.getElementById(id);


// ---------------------------------------------------------
// Availability slider
// ---------------------------------------------------------

$("availability_365").addEventListener("input", e => {
    $("availabilityValue").value = e.target.value;
});


// ---------------------------------------------------------
// Theme
// ---------------------------------------------------------

$("themeBtn").addEventListener("click", () => {
    document.body.classList.toggle("light");

    localStorage.setItem(
        "staysense-theme",
        document.body.classList.contains("light")
            ? "light"
            : "dark"
    );
});

if (localStorage.getItem("staysense-theme") === "light") {
    document.body.classList.add("light");
}


// ---------------------------------------------------------
// Example button
// ---------------------------------------------------------

$("exampleBtn").addEventListener("click", () => {

    const d = examples[exIndex++ % examples.length];

    for (const [key, value] of Object.entries(d)) {

        const element = $(key);

        if (element) {
            element.value = value;
        }
    }

    $("availabilityValue").value = d.availability_365;
});


// ---------------------------------------------------------
// Build prediction payload
// ---------------------------------------------------------

function payload() {

    const f = new FormData($("predictForm"));

    return {
        latitude: +f.get("latitude"),
        longitude: +f.get("longitude"),
        price: +f.get("price"),
        minimum_nights: +f.get("minimum_nights"),
        number_of_reviews: +f.get("number_of_reviews"),
        reviews_per_month: +f.get("reviews_per_month"),
        calculated_host_listings_count:
            +f.get("calculated_host_listings_count"),
        availability_365: +f.get("availability_365"),
        neighbourhood_group:
            f.get("neighbourhood_group"),
        neighbourhood:
            f.get("neighbourhood")
    };
}


// ---------------------------------------------------------
// POST helper
// ---------------------------------------------------------

async function jsonPost(url, data) {

    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    });

    if (!response.ok) {

        let detail = null;

        try {
            detail = await response.json();
        } catch (_) {}

        throw new Error(
            detail?.detail
                ? JSON.stringify(detail.detail)
                : `Request failed (${response.status})`
        );
    }

    return response.json();
}


// ---------------------------------------------------------
// Save report data
// IMPORTANT:
// localStorage is used because report.html opens in a new tab.
// ---------------------------------------------------------

function saveReportData() {

    if (!lastPayload || !lastExplanation) {
        return;
    }

    localStorage.setItem(
        "staysensePayload",
        JSON.stringify(lastPayload)
    );

    localStorage.setItem(
        "staysenseExplanation",
        JSON.stringify(lastExplanation)
    );
}


// ---------------------------------------------------------
// Prediction form
// ---------------------------------------------------------

$("predictForm").addEventListener("submit", async e => {

    e.preventDefault();

    $("error").textContent = "";

    if (!e.target.reportValidity()) {
        return;
    }

    lastPayload = payload();

    setLoading(true);

    try {

        // 1. Prediction
        const prediction = await jsonPost(
            PREDICT_ENDPOINT,
            lastPayload
        );

        renderPrediction(prediction);


        // 2. SHAP explanation
        const explanation = await jsonPost(
            EXPLAIN_ENDPOINT,
            lastPayload
        );

        lastExplanation = explanation;


        // 3. Render SHAP
        renderShap(explanation);


        // 4. Save history
        saveHistory(explanation);


        // 5. Save report data
        saveReportData();

    } catch (error) {

        console.error(error);

        $("error").textContent =
            error.message || "Something went wrong.";

    } finally {

        setLoading(false);
    }
});


// ---------------------------------------------------------
// Loading state
// ---------------------------------------------------------

function setLoading(value) {

    $("predictBtn").disabled = value;

    $("predictBtn").classList.toggle(
        "loading",
        value
    );
}


// ---------------------------------------------------------
// Prediction chart
// ---------------------------------------------------------

function renderPrediction(result) {

    $("empty").hidden = true;

    $("prediction").hidden = false;

    $("predictionName").textContent =
        result.Predicted_room_type;

    const maxProbability =
        Math.max(...result.Probability);

    $("confidence").textContent =
        `${(maxProbability * 100).toFixed(1)}%`;

    // Shared room risk badge (shown only when Shared room is not the
    // prediction but still has a meaningful probability).
    const riskEl = $("sharedRisk");
    if (result.Shared_room_risk) {
        riskEl.textContent =
            `⚠ Shared room risk: ${(result.Shared_room_probability * 100).toFixed(0)}% chance`;
        riskEl.hidden = false;
    } else {
        riskEl.hidden = true;
    }

    const labels = [
        "Entire home/apt",
        "Private room",
        "Shared room"
    ];

    if (probChart) {
        probChart.destroy();
    }

    probChart = new Chart(
        $("probChart"),
        {
            type: "bar",

            data: {
                labels: labels,

                datasets: [
                    {
                        data: result.Probability,

                        backgroundColor: [
                            "#4fd1c0",
                            "#c9a15a",
                            "#ff7086"
                        ],

                        borderRadius: 8
                    }
                ]
            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                indexAxis: "y",

                plugins: {

                    legend: {
                        display: false
                    },

                    tooltip: {

                        callbacks: {

                            label: context =>
                                ` ${(context.raw * 100).toFixed(1)}%`
                        }
                    }
                },

                scales: {

                    x: {

                        min: 0,
                        max: 1,

                        ticks: {

                            callback: value =>
                                `${value * 100}%`,

                            color:
                                getComputedStyle(document.body)
                                    .getPropertyValue("--muted")
                        },

                        grid: {
                            color:
                                "rgba(128,128,128,.12)"
                        }
                    },

                    y: {

                        ticks: {

                            color:
                                getComputedStyle(document.body)
                                    .getPropertyValue("--text")
                        },

                        grid: {
                            display: false
                        }
                    }
                }
            }
        }
    );
}


// ---------------------------------------------------------
// SHAP UI
// ---------------------------------------------------------

function renderShap(explanation) {

    $("explainCard").hidden = false;

    const list = $("shapList");

    list.innerHTML = "";

    const items = explanation.features || [];

    const max = Math.max(
        0.00001,
        ...items.map(
            item => Math.abs(Number(item.shap_value))
        )
    );

    items.forEach(item => {

        const div = document.createElement("div");

        div.className = "shap-item";

        const value =
            Number(item.shap_value);

        div.innerHTML = `
            <div class="shap-top">

                <span class="shap-name"></span>

                <span class="shap-value ${
                    value >= 0 ? "pos" : "neg"
                }">

                    ${
                        value >= 0 ? "+" : ""
                    }${value.toFixed(4)}

                </span>

            </div>

            <div class="muted">

                ${escapeHtml(
                    String(item.value ?? "")
                )}

            </div>

            <div class="bar">

                <i
                    class="${
                        value >= 0 ? "pos" : "neg"
                    }"
                    style="
                        width:${Math.max(
                            4,
                            Math.abs(value) / max * 100
                        )}%
                    "
                ></i>

            </div>
        `;

        div
            .querySelector(".shap-name")
            .textContent = item.label;

        list.appendChild(div);
    });
}


// ---------------------------------------------------------
// HTML escaping
// ---------------------------------------------------------

function escapeHtml(value) {

    return String(value).replace(
        /[&<>"']/g,
        character => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#039;"
        }[character])
    );
}


// ---------------------------------------------------------
// Prediction history
// ---------------------------------------------------------

function saveHistory(explanation) {

    let history = JSON.parse(
        localStorage.getItem("staysense-history") || "[]"
    );

    history.unshift({

        prediction:
            explanation.prediction,

        confidence:
            Math.max(...explanation.probabilities),

        neighbourhood:
            explanation.features?.find(
                item =>
                    item.feature === "neighbourhood"
            )?.value || ""
    });

    localStorage.setItem(
        "staysense-history",
        JSON.stringify(history.slice(0, 5))
    );

    renderHistory();
}


function renderHistory() {

    const history = JSON.parse(
        localStorage.getItem("staysense-history") || "[]"
    );

    const element = $("history");

    if (!history.length) {

        element.textContent =
            "No predictions yet.";

        return;
    }

    element.innerHTML = history.map(item => `

        <div class="history-row">

            <div>

                <b>
                    ${escapeHtml(
                        String(item.prediction)
                    )}
                </b>

                <br>

                <small>
                    ${escapeHtml(
                        String(item.neighbourhood)
                    )}
                </small>

            </div>

            <strong>
                ${(item.confidence * 100).toFixed(1)}%
            </strong>

        </div>

    `).join("");
}


$("clearHistory").addEventListener(
    "click",
    () => {

        localStorage.removeItem(
            "staysense-history"
        );

        renderHistory();
    }
);


// ---------------------------------------------------------
// Visual Report
// ---------------------------------------------------------

$("reportLink").addEventListener(
    "click",
    () => {

        // Make absolutely sure the latest data
        // is available before opening the new tab.
        saveReportData();
    }
);


// ---------------------------------------------------------
// PDF download
// ---------------------------------------------------------

$("pdfBtn").addEventListener(
    "click",
    async () => {

        if (!lastPayload) {
            return;
        }

        const button = $("pdfBtn");

        button.disabled = true;

        button.textContent =
            "Generating…";

        try {

            const response = await fetch(
                REPORT_ENDPOINT,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(lastPayload)
                }
            );

            if (!response.ok) {

                throw new Error(
                    `Report failed (${response.status})`
                );
            }

            const blob =
                await response.blob();

            const url =
                URL.createObjectURL(blob);

            const anchor =
                document.createElement("a");

            anchor.href = url;

            anchor.download =
                "staysense_prediction_report.pdf";

            document.body.appendChild(anchor);

            anchor.click();

            anchor.remove();

            URL.revokeObjectURL(url);

        } catch (error) {

            $("error").textContent =
                error.message;

        } finally {

            button.disabled = false;

            button.textContent =
                "Download PDF";
        }
    }
);


// ---------------------------------------------------------
// API health check
// ---------------------------------------------------------

async function checkApi() {

    const status =
        $("apiStatus");

    for (let i = 0; i < 12; i++) {

        try {

            const response =
                await fetch(
                    HEALTH_ENDPOINT,
                    {
                        cache: "no-store"
                    }
                );

            if (response.ok) {

                status.className =
                    "status online";

                status.innerHTML =
                    "<i></i> API connected";

                return;
            }

        } catch (_) {}

        status.className =
            "status";

        status.innerHTML =
            `<i></i> Waking API… ${i + 1}/12`;

        await new Promise(
            resolve =>
                setTimeout(resolve, 5000)
        );
    }

    status.className =
        "status offline";

    status.innerHTML =
        "<i></i> API unavailable";
}


// ---------------------------------------------------------
// Initialisation
// ---------------------------------------------------------

renderHistory();

checkApi();