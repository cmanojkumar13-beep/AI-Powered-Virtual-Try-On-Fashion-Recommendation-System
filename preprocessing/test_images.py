import cv2

person = cv2.imread("data/person.jpg")
clothing = cv2.imread("data/clothing.jpg")

if person is None:
    print("❌ Person image not found")
else:
    print("✅ Person image loaded")

if clothing is None:
    print("❌ Clothing image not found")
else:
    print("✅ Clothing image loaded")

if person is not None and clothing is not None:
    print("✅ Both images loaded successfully!")
    print("Person size:", person.shape)
    print("Clothing size:", clothing.shape)