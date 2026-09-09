import os
import sys
import cv2
import numpy as np


# Path to Silent-Face-Anti-Spoofing
SILENT_FACE_PATH = os.path.join(
    os.path.dirname(__file__),
    "Silent-Face-Anti-Spoofing-master"
)

# Add Silent-Face repository to Python path
if SILENT_FACE_PATH not in sys.path:
    sys.path.insert(0, SILENT_FACE_PATH)


from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name


class AntiSpoofing:

    def __init__(self, device_id=0):

        self.device_id = device_id

        # Anti-spoofing predictor
        self.model_test = AntiSpoofPredict(device_id)

        # Image cropper
        self.image_cropper = CropImage()

        # Model directory
        self.model_dir = os.path.join(
            SILENT_FACE_PATH,
            "resources",
            "anti_spoof_models"
        )

        # Find all .pth models
        self.model_names = [
            name
            for name in os.listdir(self.model_dir)
            if name.endswith(".pth")
        ]

        if not self.model_names:
            raise FileNotFoundError(
                f"No anti-spoofing models found in: {self.model_dir}"
            )

        print("Anti-spoofing models loaded:")

        for model_name in self.model_names:
            print(" -", model_name)

    def predict(self, image, face):

        # YuNet returns the face as a dictionary
        x = int(face["x"])
        y = int(face["y"])
        w = int(face["width"])
        h = int(face["height"])

        # Prediction array
        prediction = np.zeros((1, 3))

        # Run all anti-spoofing models
        for model_name in self.model_names:

            model_path = os.path.join(
                self.model_dir,
                model_name
            )

            h_input, w_input, model_type, scale = parse_model_name(
                model_name
            )

            param = {
                "org_img": image,
                "bbox": [x, y, w, h],
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True
            }

            if scale is None:
                param["crop"] = False

            img = self.image_cropper.crop(**param)

            prediction += self.model_test.predict(
                img,
                model_path
            )

        # Get predicted class
        label = int(np.argmax(prediction))

        # Average score across models
        score = float(
            prediction[0][label] / len(self.model_names)
        )

        # Silent-Face convention:
        # label 1 = REAL
        # other labels = FAKE
        is_real = bool(label == 1)

        return {
            "result": "REAL" if is_real else "FAKE",
            "is_real": is_real,
            "label": label,
            "score": round(score, 4)
        }