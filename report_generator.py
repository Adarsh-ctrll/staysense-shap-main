from __future__ import annotations

from io import BytesIO
from html import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether


def _pct(v):
    return f"{float(v) * 100:.1f}%"


def build_report_pdf(payload: dict, explanation: dict) -> bytes:
    out = BytesIO()
    doc = SimpleDocTemplate(
        out, pagesize=A4,
        rightMargin=16*mm, leftMargin=16*mm,
        topMargin=16*mm, bottomMargin=16*mm,
        title="StaySense Prediction Report",
        author="StaySense",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title2", parent=styles["Title"], fontSize=22, leading=26, spaceAfter=8)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, leading=18, spaceBefore=12, spaceAfter=7)
    body = ParagraphStyle("Body2", parent=styles["BodyText"], fontSize=9.5, leading=14)

    story = [
        Paragraph("StaySense", title),
        Paragraph("NYC Airbnb Room Type Prediction Report", styles["Heading3"]),
        Spacer(1, 5*mm),
        Paragraph("Prediction", h2),
        Paragraph(f"<b>{escape(str(explanation['prediction']))}</b> — {max(explanation['probabilities']):.1%} confidence", body),
    ]

    story.append(Paragraph("Class probabilities", h2))
    prob_rows = [["Room type", "Probability"]]
    for cls, p in zip(explanation["classes"], explanation["probabilities"]):
        prob_rows.append([str(cls), _pct(p)])
    table = Table(prob_rows, colWidths=[115*mm, 40*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#18233A")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#D9DEE8")),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story += [table]

    story.append(Paragraph("Listing inputs", h2))
    input_rows = [["Feature", "Value"]]
    labels = {
        "latitude":"Latitude", "longitude":"Longitude", "price":"Price per night",
        "minimum_nights":"Minimum nights", "number_of_reviews":"Total reviews",
        "reviews_per_month":"Reviews per month", "calculated_host_listings_count":"Host listing count",
        "availability_365":"Availability (days/year)", "neighbourhood_group":"Borough",
        "neighbourhood":"Neighbourhood",
    }
    for k, v in payload.items():
        input_rows.append([labels.get(k, k), str(v)])
    table2 = Table(input_rows, colWidths=[115*mm, 40*mm])
    table2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#18233A")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#D9DEE8")),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story += [table2]

    story.append(Paragraph("Why this prediction?", h2))
    story.append(Paragraph(
        "SHAP values show how each feature contributes toward or away from the predicted class. "
        "Positive values push the prediction toward the predicted class; negative values push it away.", body
    ))
    shap_rows = [["Feature", "Value", "Contribution"]]
    for item in explanation["features"]:
        shap_rows.append([
            str(item["label"]), str(item["value"]), f"{item['shap_value']:+.4f}"
        ])
    table3 = Table(shap_rows, colWidths=[70*mm, 50*mm, 35*mm])
    table3.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#18233A")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#D9DEE8")),
        ("PADDING", (0,0), (-1,-1), 5),
    ]))
    story += [table3, Spacer(1, 6*mm), Paragraph(
        f"Model explanation used {explanation['transformed_feature_count']} transformed features "
        "after the saved preprocessing pipeline.", body
    )]

    doc.build(story)
    return out.getvalue()
