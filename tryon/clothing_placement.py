import cv2
import numpy as np

from preprocessing.pose_detection import detect_pose


PERSON_PATH = "data/person.jpg"
CLOTHING_PATH = "data/clothing_no_background.png"
OUTPUT_PATH = "data/final_tryon.png"


# -----------------------------
# 1. Load images
# -----------------------------

person = cv2.imread(PERSON_PATH)
clothing = cv2.imread(CLOTHING_PATH, cv2.IMREAD_UNCHANGED)

if person is None:
    raise FileNotFoundError("❌ Person image not found.")

if clothing is None:
    raise FileNotFoundError("❌ Clothing image not found.")


# -----------------------------
# 2. Detect pose
# -----------------------------

result = detect_pose(person)

if not result.pose_landmarks:
    raise RuntimeError("❌ No person detected.")

landmarks = result.pose_landmarks[0]

height, width = person.shape[:2]


# -----------------------------
# 3. Get body landmarks
# -----------------------------

# MediaPipe:
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


# -----------------------------
# 4. Define target torso
# -----------------------------

# Expand shoulders slightly
target_top_left = LS + np.array([-25, -10], dtype=np.float32)
target_top_right = RS + np.array([25, -10], dtype=np.float32)

# Expand hips slightly
target_bottom_left = LH + np.array([-30, 15], dtype=np.float32)
target_bottom_right = RH + np.array([30, 15], dtype=np.float32)


target_points = np.array([
    target_top_left,
    target_top_right,
    target_bottom_right,
    target_bottom_left
], dtype=np.float32)


# -----------------------------
# 5. Get clothing dimensions
# -----------------------------

cloth_height, cloth_width = clothing.shape[:2]

source_points = np.array([
    [0, 0],
    [cloth_width - 1, 0],
    [cloth_width - 1, cloth_height - 1],
    [0, cloth_height - 1]
], dtype=np.float32)


# -----------------------------
# 6. Perspective transformation
# -----------------------------

matrix = cv2.getPerspectiveTransform(
    source_points,
    target_points
)


warped_clothing = cv2.warpPerspective(
    clothing,
    matrix,
    (width, height)
)


# -----------------------------
# 7. Blend clothing
# -----------------------------

if clothing.shape[2] == 4:

    alpha_original = clothing[:, :, 3]

    warped_alpha = cv2.warpPerspective(
        alpha_original,
        matrix,
        (width, height)
    )

    alpha = warped_alpha.astype(np.float32) / 255.0

else:

    # If transparency is missing,
    # create a simple mask
    gray = cv2.cvtColor(
        warped_clothing,
        cv2.COLOR_BGR2GRAY
    )

    alpha = (gray > 5).astype(np.float32)


alpha = np.expand_dims(alpha, axis=2)


# Use only RGB/BGR channels
warped_rgb = warped_clothing[:, :, :3]


# Blend
result_image = (
    warped_rgb.astype(np.float32) * alpha
    + person.astype(np.float32) * (1 - alpha)
)


result_image = np.clip(
    result_image,
    0,
    255
).astype(np.uint8)


# -----------------------------
# 8. Save result
# -----------------------------

cv2.imwrite(
    OUTPUT_PATH,
    result_image
)


print()
print("✅ Improved virtual try-on created!")
print(f"Saved to: {OUTPUT_PATH}")