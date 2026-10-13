"""
Feature engineering for StaySense.

Kept in its own module so the pickled pipeline (model_pipeline.pkl) can be
loaded by main_shap.py: pickle stores a reference to this class, so this file
must sit next to the notebook and next to main_shap.py.

The API still takes the same 10 raw inputs. Everything new is derived from
those inputs plus statistics learned from the TRAINING data only.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TIMES_SQUARE = (40.7580, -73.9855)
MIN_HOOD_LISTINGS = 5  # fewer listings than this -> fall back to borough median

NEW_FEATURES = [
    "price_vs_hood_median",     # price / median price of the same neighbourhood
    "price_vs_borough_median",  # price / median price of the same borough
    "dist_times_square_km",     # distance to Times Square
    "hood_listing_count",       # how many training listings the neighbourhood has
    "has_reviews",              # 1 if the listing has any review
    "is_multi_host",            # 1 if host has more than one listing
    "is_always_available",      # 1 if available (almost) every day of the year
]


def _haversine_km(lat, lon, lat0, lon0):
    lat, lon = np.radians(lat.astype(float)), np.radians(lon.astype(float))
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    a = (np.sin((lat - lat0) / 2) ** 2
         + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


class ListingFeatures(BaseEstimator, TransformerMixin):
    """Adds engineered columns to the raw listing DataFrame."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.input_columns_ = list(X.columns)
        self.global_median_ = float(X["price"].median())
        self.borough_median_ = X.groupby("neighbourhood_group")["price"].median().to_dict()

        counts = X["neighbourhood"].value_counts()
        self.hood_count_ = counts.to_dict()
        big = counts[counts >= MIN_HOOD_LISTINGS].index
        self.hood_median_ = (
            X[X["neighbourhood"].isin(big)]
            .groupby("neighbourhood")["price"].median().to_dict()
        )
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        borough_med = (X["neighbourhood_group"].map(self.borough_median_)
                       .fillna(self.global_median_).clip(lower=1))
        hood_med = (X["neighbourhood"].map(self.hood_median_)
                    .fillna(borough_med).clip(lower=1))

        X["price_vs_hood_median"] = X["price"] / hood_med
        X["price_vs_borough_median"] = X["price"] / borough_med
        X["dist_times_square_km"] = _haversine_km(
            X["latitude"].to_numpy(), X["longitude"].to_numpy(), *TIMES_SQUARE
        )
        X["hood_listing_count"] = X["neighbourhood"].map(self.hood_count_).fillna(0)
        X["has_reviews"] = (X["number_of_reviews"] > 0).astype(int)
        X["is_multi_host"] = (X["calculated_host_listings_count"] > 1).astype(int)
        X["is_always_available"] = (X["availability_365"] >= 360).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.input_columns_ + NEW_FEATURES, dtype=object)