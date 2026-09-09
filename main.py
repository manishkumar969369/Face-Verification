from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np

from detector import YuNetDetector
from blur_checker import BlurChecker
from anti_spoofing import AntiSpoofing


# =========================================================
# FastAPI App
# =========================================================

app = FastAPI(
    title="Face Verification API",
    description="Face detection, blur detection and anti-spoofing API",
    version="1.0.0"
)


# =========================================================
# Model Paths
# =========================================================

MODEL_PATH = "models/face_detection_yunet.onnx"


# =========================================================
# Initialize Models
# =========================================================

detector = YuNetDetector(MODEL_PATH)

blur_checker = BlurChecker(
    threshold=100.0,
    margin=30
)

anti_spoofing = AntiSpoofing()


# =========================================================
# Root Endpoint
# =========================================================

@app.get("/")
def home():
    return {
        "message": "Face Verification API is running",
        "endpoint": "POST /detect-face"
    }


# =========================================================
# Face Detection + Blur + Anti-Spoofing
# =========================================================

@app.post("/detect-face")
async def detect_face(file: UploadFile = File(...)):

    # -----------------------------------------------------
    # 1. Check uploaded file type
    # -----------------------------------------------------

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="File type is missing"
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image"
        )


    # -----------------------------------------------------
    # 2. Read image
    # -----------------------------------------------------

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )


    # Convert bytes to NumPy array
    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )


    # Decode image using OpenCV
    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image"
        )


    # -----------------------------------------------------
    # 3. Detect face using YuNet
    # -----------------------------------------------------

    faces = detector.detect(image)


    # -----------------------------------------------------
    # 4. No face detected
    # -----------------------------------------------------

    if len(faces) == 0:

        return {
            "success": False,
            "result": "NO_FACE",
            "message": "No face detected in the image. Reupload image",
            "faces_detected": 0
        }


    # -----------------------------------------------------
    # 5. Use first detected face
    # -----------------------------------------------------

    face = faces[0]


    # -----------------------------------------------------
    # 6. Blur Detection
    # -----------------------------------------------------

    blur_result = blur_checker.check(
        image,
        face
    )


    # -----------------------------------------------------
    # 7. If image is blurry
    # -----------------------------------------------------

    if blur_result["is_blurry"]:

        return {
            "success": True,
            "result": "BLURRY",
            "message": "Image is blurry. Please upload a clear image.",
            "faces_detected": len(faces),
            "face": face,
            "blur_check": blur_result
        }


    # -----------------------------------------------------
    # 8. Anti-Spoofing
    # -----------------------------------------------------

    spoof_result = anti_spoofing.predict(
        image,
        face
    )


    # -----------------------------------------------------
    # 9. REAL / FAKE
    # -----------------------------------------------------

    if spoof_result["result"] == "FAKE":

        return {
            "success": True,
            "result": "FAKE",
            "message": "Fake or sblurry face detected.",
            "faces_detected": len(faces),
            "face": face,
            "blur_check": blur_result,
            "anti_spoofing": spoof_result
        }


    # -----------------------------------------------------
    # 10. REAL face
    # -----------------------------------------------------

    return {
        "success": True,
        "result": "REAL",
        "message": "Real face detected.",
        "faces_detected": len(faces),
        "face": face,
        "blur_check": blur_result,
        "anti_spoofing": spoof_result
    }