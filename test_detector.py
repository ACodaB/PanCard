from app.services.detector import Detector
detector = Detector(
    model_path="C:\\Coding\\Decimal Internship\\Pan card\\runs\\detect\\runs\\pan_card_yolo12s-9\\weights\\best.pt",
    conf_threshold=0.8
)

print("Detector initialized successfully!")