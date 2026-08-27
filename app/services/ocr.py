
import cv2
from paddleocr import PaddleOCR

class OCR:
    def __init__(self):
        self.ocr = PaddleOCR(
            lang="en",
            use_textline_orientation=True
        )

        print("PaddleOCR loaded.")

    def extract_text(self, crops: list):

        results = []

        for crop in crops:
            img=crop["image"]

            img = cv2.resize(
                img,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
            )

            img = cv2.copyMakeBorder(
                img,
                top=20,
                bottom=20,
                left=20,
                right=20,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255)   # White border
            )

            ocr_result = self.ocr.predict(img)

            # ocr_result = self.ocr.predict(crop["image"])

            text = ""
            confidence = 0.0

            if len(ocr_result) > 0:

                rec_texts = ocr_result[0].get("rec_texts", [])
                rec_scores = ocr_result[0].get("rec_scores", [])

                if rec_texts:
                    text = " ".join(rec_texts)

                if rec_scores:
                    confidence = max(rec_scores)

            results.append(
                {
                    "class": crop["class"],
                    "text": text,
                    "confidence": confidence
                }
            )

        return results
    
