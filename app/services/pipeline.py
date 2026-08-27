from app.services.detector import Detector
from app.services.cropper import Cropper
from app.services.ocr import OCR
from app.services.parser import Parser


class Pipeline:

    def __init__(self, model_path):

        self.detector = Detector(model_path)
        self.cropper = Cropper(padding=12)  
        self.ocr = OCR()
        self.parser = Parser()

    def predict(self, image):

        detections = self.detector.detect(image)

        crops = self.cropper.crop(image, detections)

        ocr_results = self.ocr.extract_text(crops)

        result = self.parser.parse(ocr_results)

        return result