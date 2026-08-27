from ultralytics import YOLO
import numpy as np
class Detector:
    def __init__(self, model_path : str , conf_threshold : float = 0.6):
        """
        Initialize the detector.

        Parameters
        ----------
        model_path : str
            Path to trained YOLO model.

        conf_threshold : float
            Minimum confidence required to keep a detection.
        """

        self.conf_threshold = conf_threshold

        self.model = YOLO(model_path)

        print(f"YOLO model loaded from {model_path}")
        
    def detect(self, image : np.ndarray):
        if image is None:
            raise ValueError("Input image is None.")

        results = self.model(image)

        result = results[0]

        detections = []

        for box in result.boxes:

            confidence = box.conf.item()

            if confidence < self.conf_threshold:
                continue

            class_id = int(box.cls.item())
            class_name = result.names[class_id]

            x1, y1, x2, y2 = (
                box.xyxy
                .cpu()
                .numpy()[0]
                .astype(int)
            )

            detections.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                }
            )

        return detections