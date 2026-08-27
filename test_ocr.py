from app.services.ocr import OCR
from app.services.cropper import Cropper
from app.services.detector import Detector
import cv2

image = cv2.imread(r"C:\Coding\Decimal Internship\pan_ocr_api\test_images_for_pipeline\59806e62-58d8e0b0ca35d5ce7347b5d9_pan_jpg.rf.489c5092e35155a538c7b6a65d0ee4d1.jpg")
# image2 = cv2.imread(r"C:\Users\arsh2\Desktop\Internship\PAN Card.jpg")

detector = Detector(r"C:\Coding\Decimal Internship\Pan card\runs\detect\runs\pan_card_yolo12s-10\weights\best.pt")
cropper = Cropper(padding=12)
ocr = OCR()

detections = detector.detect(image)
# detections2 = detector.detect(image2)


crops = cropper.crop(image, detections)
for crop in crops:
    cv2.imwrite(f"{crop['class']}_cropptest.jpg", crop["image"])

# crops2 = cropper.crop(image2, detections2)
# for crop in crops2:
#     cv2.imwrite(f"{crop['class']}_cropptest.jpg", crop["image"])

result = ocr.extract_text(crops)
# result2 = ocr.extract_text(crops2)

print(result)
# print(result2)