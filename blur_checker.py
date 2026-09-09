import cv2


class BlurChecker:
    def __init__(self, threshold=100.0, margin=30):
        self.threshold = threshold
        self.margin = margin

    def check(self, image, face):

        # Face is a dictionary returned by YuNetDetector
        x = face["x"]
        y = face["y"]
        w = face["width"]
        h = face["height"]

        # Add some margin around the face
        x1 = max(0, x - self.margin)
        y1 = max(0, y - self.margin)

        x2 = min(image.shape[1], x + w + self.margin)
        y2 = min(image.shape[0], y + h + self.margin)

        # Crop face region
        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            return {
                "is_blurry": True,
                "blur_score": 0.0,
                "threshold": self.threshold,
                "message": "Invalid face crop"
            }

        # Convert to grayscale
        gray_face = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2GRAY
        )

        # Calculate Laplacian variance
        blur_score = cv2.Laplacian(
            gray_face,
            cv2.CV_64F
        ).var()

        blur_score = float(blur_score)

        # IMPORTANT: convert numpy.bool_ to Python bool
        is_blurry = bool(
            blur_score < self.threshold
        )

        return {
            "is_blurry": is_blurry,
            "blur_score": round(blur_score, 2),
            "threshold": self.threshold
        }