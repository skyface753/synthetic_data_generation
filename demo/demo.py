import streamlit as st
import os
import shutil
import json
from PIL import Image
from SynDataGenYOLO.extract import extract_objects_from_labelme_data
from SynDataGenYOLO.data_generation import SyntheticImageGenerator, BlendingMode, OutputMode
from pathlib import Path

# --- Setup Directories and Predefined Images ---
PREDEFINED_DIR = Path("demo_data")
PREDEFINED_FOREGROUNDS_DIR = PREDEFINED_DIR / "foregrounds"
PREDEFINED_BACKGROUNDS_DIR = PREDEFINED_DIR / "backgrounds"

# Ensure directories exist
for p in [PREDEFINED_FOREGROUNDS_DIR, PREDEFINED_BACKGROUNDS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.JPG')

# Predefined images, now allowing both png and jpg
predefined_foregrounds = sorted([f for f in os.listdir(PREDEFINED_FOREGROUNDS_DIR)
                                 if f.endswith(IMAGE_EXTENSIONS)])
predefined_backgrounds = sorted([f for f in os.listdir(PREDEFINED_BACKGROUNDS_DIR)
                                 if f.endswith(IMAGE_EXTENSIONS)])


# --- Helper Functions ---


def run_extraction(input_path, output_path, margin=20):
    """
    Wrapper to run the object extraction.
    This will create a 'foregrounds' directory inside output_path.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    extract_objects_from_labelme_data(input_path, output_path, margin)

# Function to get the file path for a given filename


def get_foreground_path(filename):
    return str(PREDEFINED_FOREGROUNDS_DIR / filename)


def get_background_path(filename):
    return str(PREDEFINED_BACKGROUNDS_DIR / filename)


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("Synthetic Data Generation Demo")

# Custom CSS for image selection
st.markdown("""
<style>
    .selected-image-container {
        border: 3px solid #4CAF50; /* Highlight color */
        border-radius: 5px;
        padding: 5px;
    }
    .image-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
    }
    .stImage > img {
        border-radius: 5px;
        object-fit: cover;
    }
</style>
""", unsafe_allow_html=True)


col1, col2 = st.columns(2)

# Initialize session state for selections
if 'selected_foreground' not in st.session_state:
    st.session_state.selected_foreground = None
if 'selected_background' not in st.session_state:
    st.session_state.selected_background = None

with col1:
    st.markdown("### Foreground Objects")
    st.write("Click an image to select it.")
    foreground_cols = st.columns(3)
    for i, fg_file in enumerate(predefined_foregrounds):
        with foreground_cols[i % 3]:
            # Use a button to make the image selectable
            is_selected = st.session_state.selected_foreground == fg_file
            container_style = "selected-image-container" if is_selected else "image-container"

            st.markdown(
                f'<div class="{container_style}">', unsafe_allow_html=True)
            if st.button(label=f"_{fg_file}_", key=f"fg_{i}", use_container_width=True):
                st.session_state.selected_foreground = fg_file
                st.rerun()  # Rerun to update the UI
            st.image(get_foreground_path(fg_file), caption=fg_file, width=150)
            st.markdown('</div>', unsafe_allow_html=True)


with col2:
    st.markdown("### Backgrounds")
    st.write("Click an image to select it.")
    background_cols = st.columns(3)
    for i, bg_file in enumerate(predefined_backgrounds):
        with background_cols[i % 3]:
            is_selected = st.session_state.selected_background == bg_file
            container_style = "selected-image-container" if is_selected else "image-container"

            st.markdown(
                f'<div class="{container_style}">', unsafe_allow_html=True)
            if st.button(label=f"_{bg_file}_", key=f"bg_{i}", use_container_width=True):
                st.session_state.selected_background = bg_file
                st.rerun()
            st.image(get_background_path(bg_file), caption=bg_file, width=150)
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

if st.button("Generate Synthetic Image", use_container_width=True):
    if st.session_state.selected_foreground and st.session_state.selected_background:
        # Define temporary directories for processing
        temp_extracted_dir = "temp_extracted"
        temp_output_dir = "synthetic_output"

        # Clean up previous runs
        if os.path.exists(temp_extracted_dir):
            shutil.rmtree(temp_extracted_dir)
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)

        st.info(
            f"Extracting object from {st.session_state.selected_foreground}...")
        foreground_path = PREDEFINED_FOREGROUNDS_DIR / \
            st.session_state.selected_foreground
        # Assuming the labelme JSON file has the same name as the image (e.g., image.png, image.json)
        # and is in the same directory.
        run_extraction(str(foreground_path.parent),
                       temp_extracted_dir, margin=20)

        st.info("Generating synthetic image...")
        gen_input_dir = Path("gen_input")
        if gen_input_dir.exists():
            shutil.rmtree(gen_input_dir)
        os.makedirs(gen_input_dir)

        # Copy extracted objects and selected background into the input structure
        shutil.copytree(temp_extracted_dir, gen_input_dir / "foregrounds")
        os.makedirs(gen_input_dir / "backgrounds")
        shutil.copy(PREDEFINED_BACKGROUNDS_DIR /
                    st.session_state.selected_background, gen_input_dir / "backgrounds")

        # Run the generator
        data_generator = SyntheticImageGenerator(
            input_dir=str(gen_input_dir),
            output_dir=temp_output_dir,
            image_number=1,
            max_objects_per_image=1,
            image_width=600,
            image_height=400,
            fixed_image_sizes=False,
            parallelize=False,
            augmentation_path="",
            color_harmon_alpha=0.5,
            color_harmonization=False,
            avoid_collisions=True,
            debug=False,
            distractor_objects=[],
            output_mode=OutputMode.YOLO,
            scale_foreground_by_background_size=True,
            scaling_factors=(0.25, 0.85),
            yolo_input=False,
            random_color_harmon_alpha=False,
            pyramid_blending_levels=5,
            blending_methods=[BlendingMode.ALPHA_BLENDING,
                              BlendingMode.GAUSSIAN_BLUR],
            gaussian_options=[9, 9]
        )
        data_generator.generate_images()

        # Display the result
        st.success("Image generated!")
        result_path = Path(temp_output_dir) / "images"
        if os.listdir(result_path):
            st.image(str(result_path / os.listdir(result_path)
                     [0]), caption="Generated Synthetic Image")

        # Clean up
        shutil.rmtree(temp_extracted_dir)
        shutil.rmtree(temp_output_dir)
        shutil.rmtree(gen_input_dir)
    else:
        st.warning(
            "Please select both a foreground and a background image to generate.")
