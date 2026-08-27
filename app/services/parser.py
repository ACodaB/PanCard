class Parser:

    def parse(self, ocr_results: list):
        """
        Convert OCR output into final JSON.
        """
        output = {}

        

        for item in ocr_results:
            output[item["class"]] = {
                "text": item["text"],
                "confidence": round(item["confidence"], 6)
            }

        return output