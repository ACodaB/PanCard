from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np

from app.services.pipeline import Pipeline

app = FastAPI(
    title="PAN OCR API",
    version="1.0.0"
)

pipeline = Pipeline("app/models/best.pt")


@app.get("/")
def home():
    return {
        "message": "PAN OCR API is running."
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )

    # Read uploaded image
    contents = await file.read()

    image = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not read image."
        )

    # Run pipeline
    result = pipeline.predict(image)

    return result