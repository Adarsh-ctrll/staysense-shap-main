from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field
import joblib
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
app=FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COLUMNS = ["latitude", "longitude", "price", "minimum_nights",
    "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
    "neighbourhood_group", "neighbourhood",]

model = joblib.load("model_pipeline.pkl") 

#Pydantic Model=>input validation
class Features(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    price: float = Field(..., gt=0, description="Price per night, must be positive")
    minimum_nights: int = Field(..., ge=1, le=365, description="Minimum nights required for booking")
    number_of_reviews: int = Field(..., ge=0, description="Total number of reviews")
    reviews_per_month: float = Field(..., ge=0, description="Average reviews per month")
    calculated_host_listings_count: int = Field(..., ge=0, description="Number of listings by this host")
    availability_365: int = Field(..., ge=0, le=365, description="Days available out of 365")
    neighbourhood_group: str = Field(..., min_length=1, description="Borough or neighbourhood group")
    neighbourhood: str = Field(..., min_length=1, description="Specific neighbourhood name")




@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/style.css")
def css():
    return FileResponse("style.css", media_type="text/css")


@app.get("/script.js")
def javascript():
    return FileResponse("script.js", media_type="application/javascript")


# @app.get("/")
# def greet():
#     return "Hello Guys"

@app.post("/predict")
def predict(features: Features):

    row = pd.DataFrame(
        [features.model_dump()],
        columns=COLUMNS
    )

    print(row)
    print(row.dtypes)
    print(row.isna().sum())

    print(
        "Has infinity:",
        row.select_dtypes(include="number")
        .isin([float("inf"), float("-inf")])
        .any()
        .any()
    )

    prediction = model.predict(row)
    probability = model.predict_proba(row)

    return {
        "Predicted_room_type": prediction[0],
        "Probability": probability.tolist()[0]
    }