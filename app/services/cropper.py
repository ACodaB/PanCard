import cv2
import numpy as np

class Cropper:
    def __init__(self, padding=5):
        """
        Initialize cropper.

        Parameters
        ----------
        padding : int
            Extra pixels to add around each bounding box.
        """
        self.padding = padding

    def crop(self, image: np.ndarray, detections: list):
        """
        Crop detected regions from an image.

        Parameters
        ----------
        image : np.ndarray
            Original image.

        detections : list
            Output from detector.detect()

        Returns
        -------
        list
            [
                {
                    "class": "name",
                    "image": cropped_image
                },
                ...
            ]
        """

        if image is None:
            raise ValueError("Input image is None.")

        height, width = image.shape[:2]

        crops = []

        for detection in detections:

            x1, y1, x2, y2 = detection["bbox"]

            # Add padding
            x1 -= self.padding
            y1 -= self.padding
            x2 += self.padding
            y2 += self.padding

            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]

            crops.append(
                {
                    "class": detection["class"],
                    "image": crop
                }
            )

        return crops