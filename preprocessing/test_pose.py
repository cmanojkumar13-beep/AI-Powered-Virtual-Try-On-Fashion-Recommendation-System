import cv2

from pose_detection import detect_pose


IMAGE_PATH = "data/person.jpg"


image = cv2.imread(IMAGE_PATH)

if image is None:
    print("❌ Could not find the image.")
    print("Make sure the image is inside data/person.jpg")
else:

    result = detect_pose(image)

    if result.pose_landmarks:

        print("✅ Person detected!")

        print(
            "Number of detected poses:",
            len(result.pose_landmarks)
        )

        print(
            "Number of body landmarks:",
            len(result.pose_landmarks[0])
        )

    else:

        print("❌ No person detected.")