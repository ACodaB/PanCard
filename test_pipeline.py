import cv2
import os

from app.services.pipeline import Pipeline

pipeline = Pipeline(r"C:\Coding\Decimal Internship\Pan card\runs\detect\runs\pan_card_yolo12s-10\weights\best.pt")

dir_path="test_images_for_pipeline"
for filename in os.listdir(dir_path):
    filepath=os.path.join(dir_path,filename)

    image = cv2.imread(
        filepath
    )

    result = pipeline.predict(image)

    print(result,end="\n\n")