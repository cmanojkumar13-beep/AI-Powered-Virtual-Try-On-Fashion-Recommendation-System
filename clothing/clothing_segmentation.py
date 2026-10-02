from rembg import remove, new_session
from PIL import Image


input_path = "data/clothing.jpg"
output_path = "data/clothing_no_background.png"


print("Loading clothing image...")

input_image = Image.open(input_path)

print("Removing background...")

# Use the lightweight U2Net model
session = new_session("u2netp")

output_image = remove(
    input_image,
    session=session
)

output_image.save(output_path)

print("✅ Clothing background removed!")
print(f"Saved to: {output_path}")