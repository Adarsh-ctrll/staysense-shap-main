from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import shap

COLUMNS = [
    "latitude", "longitude", "price", "minimum_nights",
    "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
    "neighbourhood_group", "neighbourhood",
]

NUMERIC_LABELS = {
    "latitude": "Latitude",
    "longitude": "Longitude",
    "price": "Price per night",
    "minimum_nights": "Minimum nights",
    "number_of_reviews": "Total reviews",
    "reviews_per_month": "Reviews per month",
    "calculated_host_listings_count": "Host listing count",
    "availability_365": "Availability (days/year)",
}

CATEGORICAL_LABELS = {
    "neighbourhood_group": "Borough",
    "neighbourhood": "Neighbourhood",
}


def _pipeline_parts(model):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    return preprocessor, classifier


@lru_cache(maxsize=4)
def _explainer(classifier):
    return shap.TreeExplainer(classifier)


def _feature_names(preprocessor) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def _class_axis(values: Any, n_classes: int, n_features: int):
    """Normalize SHAP output to (n_samples, n_classes, n_features)."""
    if isinstance(values, list):
        arrs = [np.asarray(v) for v in values]
        return np.stack(arrs, axis=1)

    arr = np.asarray(values)
    if arr.ndim == 2:
        # Binary classifiers can return (samples, features).
        if n_classes == 2:
            return np.stack([-arr, arr], axis=1)
        raise ValueError(f"Unexpected 2-D SHAP output for {n_classes} classes: {arr.shape}")

    if arr.ndim != 3:
        raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")

    # SHAP versions can expose either (samples, features, classes)
    # or (samples, classes, features).
    if arr.shape[1] == n_features and arr.shape[2] == n_classes:
        return np.transpose(arr, (0, 2, 1))
    if arr.shape[1] == n_classes and arr.shape[2] == n_features:
        return arr

    raise ValueError(
        f"Cannot identify class/feature axes in SHAP output {arr.shape}; "
        f"expected features={n_features}, classes={n_classes}."
    )


def _expected_value(explainer, class_index: int, n_classes: int) -> float | None:
    base = explainer.expected_value
    if base is None:
        return None
    arr = np.asarray(base, dtype=float).reshape(-1)
    if arr.size == 1:
        return float(arr[0])
    if class_index < arr.size:
        return float(arr[class_index])
    return None


def explain_row(model, row: pd.DataFrame, top_n: int = 8) -> dict:
    preprocessor, classifier = _pipeline_parts(model)
    transformed = preprocessor.transform(row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=np.float32)

    names = _feature_names(preprocessor)
    n_features = len(names)
    classes = list(classifier.classes_)
    prediction = classifier.predict(transformed)[0]
    class_index = classes.index(prediction)

    explainer = _explainer(classifier)
    raw_values = explainer.shap_values(transformed)
    values = _class_axis(raw_values, len(classes), n_features)
    class_values = values[0, class_index, :]

    # Group one-hot columns back to the original 10 user-facing features.
    grouped: dict[str, dict] = {}
    original_values = row.iloc[0].to_dict()
    for idx, transformed_name in enumerate(names):
        if transformed_name.startswith(("num__", "geo__")):
            original = transformed_name.split("__", 1)[1]
        elif transformed_name.startswith("cat__"):
            rest = transformed_name[len("cat__"):]
            original = next(
                c for c in sorted(CATEGORICAL_LABELS, key=len, reverse=True)
                if rest.startswith(c + "_")
            )
        else:
            original = transformed_name.split("__", 1)[-1]

        if original not in grouped:
            grouped[original] = {
                "feature": original,
                "label": NUMERIC_LABELS.get(original, CATEGORICAL_LABELS.get(original, original)),
                "value": original_values.get(original),
                "shap_value": 0.0,
            }
        grouped[original]["shap_value"] += float(class_values[idx])

    explanations = sorted(
        grouped.values(), key=lambda x: abs(x["shap_value"]), reverse=True
    )
    explanations = explanations[:top_n]

    # Positive/negative split is with respect to the predicted class.
    positive = sorted(
        [x for x in explanations if x["shap_value"] >= 0],
        key=lambda x: x["shap_value"], reverse=True,
    )
    negative = sorted(
        [x for x in explanations if x["shap_value"] < 0],
        key=lambda x: x["shap_value"],
    )

    probabilities = classifier.predict_proba(transformed)[0]
    return {
        "prediction": prediction,
        "predicted_class_index": class_index,
        "classes": [str(x) for x in classes],
        "probabilities": [float(x) for x in probabilities],
        "base_value": _expected_value(explainer, class_index, len(classes)),
        "features": explanations,
        "positive_features": positive,
        "negative_features": negative,
        "transformed_feature_count": n_features,
    }
