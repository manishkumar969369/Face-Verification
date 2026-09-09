import cv2


class YuNetDetector:

    def __init__(self, model_path):
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            0.6,       # score threshold
            0.20,      # NMS threshold
            5000
        )

    def detect(self, image):

        height, width = image.shape[:2]

        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(image)

        if faces is None:
            return []

        results = []

        for face in faces:
            x, y, w, h = face[:4]
            confidence = float(face[-1])

            results.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "confidence": round(confidence, 4)
            })

        return results