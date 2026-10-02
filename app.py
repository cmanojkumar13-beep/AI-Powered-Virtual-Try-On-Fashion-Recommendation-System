import streamlit as st
from pathlib import Path
from PIL import Image
from gradio_client import Client, handle_file


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

PERSON_PATH = PROJECT_DIR / "data" / "person.jpg"
CLOTHES_DIR = PROJECT_DIR / "data" / "clothes"
OUTPUT_DIR = PROJECT_DIR / "vton" / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STREAMLIT SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI Virtual Try-On",
    page_icon="👕",
    layout="wide"
)

st.title("👕 AI-Powered Virtual Try-On")
st.write("Select a clothing item and generate a virtual try-on result.")


# ============================================================
# CHECK PERSON IMAGE
# ============================================================

if not PERSON_PATH.exists():
    st.error(f"Person image not found: {PERSON_PATH}")
    st.stop()

person = Image.open(PERSON_PATH).convert("RGB")


# ============================================================
# CHECK CLOTHES
# ============================================================

if not CLOTHES_DIR.exists():
    st.error(f"Clothes folder not found: {CLOTHES_DIR}")
    st.stop()

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

clothes = sorted(
    [
        file
        for file in CLOTHES_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in extensions
    ]
)


if not clothes:
    st.error("No clothing images found.")
    st.stop()


# ============================================================
# PERSON IMAGE
# ============================================================

st.header("1. Person")

st.image(
    person,
    caption="Original Person",
    width=450
)


# ============================================================
# CLOTHING SELECTION
# ============================================================

st.header("2. Select Top")

clothing_names = [
    file.name
    for file in clothes
]

selected_name = st.selectbox(
    "Choose a top:",
    clothing_names
)

selected_cloth_path = CLOTHES_DIR / selected_name

selected_cloth = Image.open(
    selected_cloth_path
).convert("RGB")


col1, col2 = st.columns(2)

with col1:

    st.image(
        selected_cloth,
        caption=f"Selected: {selected_name}",
        width=450
    )

with col2:

    st.info(
        """
        The AI will replace the person's
        upper-body clothing with the
        selected garment.
        """
    )


# ============================================================
# TRY ON BUTTON
# ============================================================

if st.button(
    "👕 TRY ON",
    type="primary",
    use_container_width=True
):

    st.info(
        "Connecting to the AI Virtual Try-On model..."
    )

    try:

        with st.spinner(
            "Generating virtual try-on..."
        ):

            # Connect to IDM-VTON
            client = Client(
                "yisol/IDM-VTON"
            )

            # ------------------------------------------------
            # PERSON IMAGE
            # ------------------------------------------------

            person_input = {
                "background": handle_file(
                    str(PERSON_PATH)
                ),
                "layers": [],
                "composite": None
            }

            # ------------------------------------------------
            # GARMENT DESCRIPTION
            # ------------------------------------------------

            garment_description = (
                "A complete upper body shirt/top, "
                "covering the shoulders, chest, "
                "torso and upper body, "
                "realistic fabric and natural fit"
            )

            # ------------------------------------------------
            # RUN IDM-VTON
            # ------------------------------------------------

            result = client.predict(

                dict=person_input,

                garm_img=handle_file(
                    str(selected_cloth_path)
                ),

                garment_des=garment_description,

                # Automatic upper-body mask
                is_checked=True,

                # Keep complete original image
                is_checked_crop=False,

                # Quality
                denoise_steps=40,

                # Reproducible result
                seed=42,

                api_name="/tryon"
            )

        # ====================================================
        # RESULT
        # ====================================================

        if result is None:

            st.error(
                "No result was returned by the VTON model."
            )

        else:

            # IDM-VTON returns:
            # result[0] = generated image
            # result[1] = mask

            result_image = result[0]

            st.success(
                "✅ Virtual try-on completed!"
            )

            st.header("3. Final Try-On Result")

            st.image(
                result_image,
                caption=f"Person wearing {selected_name}",
                use_container_width=True
            )

            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            output_path = (
                OUTPUT_DIR /
                f"tryon_{selected_name}.png"
            )

            try:

                result_image.save(
                    output_path
                )

                st.success(
                    f"Result saved to: {output_path}"
                )

            except Exception as save_error:

                st.warning(
                    f"Could not save result: {save_error}"
                )


    except Exception as error:

        st.error(
            "Virtual try-on failed."
        )

        st.code(
            str(error)
        )


# ============================================================
# CLOTHING GALLERY
# ============================================================

st.divider()

st.header("4. Other Tops")

gallery_columns = st.columns(4)

for index, cloth_file in enumerate(clothes):

    with gallery_columns[index % 4]:

        image = Image.open(
            cloth_file
        ).convert("RGB")

        st.image(
            image,
            caption=cloth_file.name,
            use_container_width=True
        )