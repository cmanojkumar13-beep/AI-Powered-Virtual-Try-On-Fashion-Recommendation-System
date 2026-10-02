import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/pose/pose_landmarker.task"


# Create Pose Landmarker
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)

landmarker = vision.PoseLandmarker.create_from_options(options)


def detect_pose(image):
    """
    Detect human body landmarks.
    """

    # OpenCV BGR → RGB
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Convert OpenCV image to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    # Detect pose
    result = landmarker.detect(mp_image)

    return result