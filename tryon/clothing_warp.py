import cv2
import numpy as np

from preprocessing.pose_detection import detect_pose


PERSON_PATH = "data/person.jpg"
CLOTHING_PATH = "data/clothing_no_background.png"
OUTPUT_PATH = "data/final_tryon.png"


# Load person image
person = cv2.imread(PERSON_PATH)

# Load clothing image with transparency
clothing = cv2.imread(
    CLOTHING_PATH,
    cv2.IMREAD_UNCHANGED
)

if person is None:
    raise FileNotFoundError("Person image not found.")

if clothing is None:
    raise FileNotFoundError("Clothing image not found.")


# Detect pose
result = detect_pose(person)

if not result.pose_landmarks:
    raise RuntimeError("No person detected.")

landmarks = result.pose_landmarks[0]

height, width = person.shape[:2]


# MediaPipe landmarks
# 11 = left shoulder
# 12 = right shoulder
# 23 = left hip
# 24 = right hip

left_shoulder = landmarks[11]
right_shoulder = landmarks[12]
left_hip = landmarks[23]
right_hip = landmarks[24]


def to_pixel(landmark):
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return np.array([x, y], dtype=np.float32)


LS = to_pixel(left_shoulder)
RS = to_pixel(right_shoulder)
LH = to_pixel(left_hip)
RH = to_pixel(right_hip)


print("Left shoulder:", LS)
print("Right shoulder:", RS)
print("Left hip:", LH)
print("Right hip:", RH)


# Target area on the person's torso
target_points = np.array([
    LS + [-30, -10],
    RS + [30, -10],
    RH + [35, 20],
    LH + [-35, 20]
], dtype=np.float32)


# Clothing dimensions
cloth_height, cloth_width = clothing.shape[:2]

source_points = np.array([
    [0, 0],
    [cloth_width - 1, 0],
    [cloth_width - 1, cloth_height - 1],
    [0, cloth_height - 1]
], dtype=np.float32)


# Perspective transformation
matrix = cv2.getPerspectiveTransform(
    source_points,
    target_points
)


warped_clothing = cv2.warpPerspective(
    clothing,
    matrix,
    (width, height)
)


# Create transparency mask
if clothing.shape[2] == 4:

    alpha_original = clothing[:, :, 3]

    warped_alpha = cv2.warpPerspective(
        alpha_original,
        matrix,
        (width, height)
    )

    alpha = warped_alpha.astype(np.float32) / 255.0

else:

    gray = cv2.cvtColor(
        warped_clothing,
        cv2.COLOR_BGR2GRAY
    )

    alpha = (gray > 5).astype(np.float32)


alpha = np.expand_dims(alpha, axis=2)


# Clothing RGB channels
warped_rgb = warped_clothing[:, :, :3]


# Blend clothing with person
result_image = (
    warped_rgb.astype(np.float32) * alpha
    + person.astype(np.float32) * (1 - alpha)
)


result_image = np.clip(
    result_image,
    0,
    255
).astype(np.uint8)


# Save result
cv2.imwrite(
    OUTPUT_PATH,
    result_image
)


print()
print("✅ Improved virtual try-on created!")
print(f"Saved to: {OUTPUT_PATH}")