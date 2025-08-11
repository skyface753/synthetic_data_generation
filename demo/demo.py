import os
import shutil
from pathlib import Path

import streamlit as st

from syndatagenyolo.data_generation import (
    BlendingMode,
    ScalingConfig,
    BlendingConfig,
    OutputMode,
    SyntheticImageGenerator,
)
from syndatagenyolo.extract import extract_objects_from_labelme_data

# --- Setup Directories and Predefined Images ---
PREDEFINED_DIR = "demo_data"
PREDEFINED_FOREGROUNDS_DIR = os.path.join(
    os.path.dirname(__file__), PREDEFINED_DIR, "foregrounds"
)
PREDEFINED_BACKGROUNDS_DIR = os.path.join(
    os.path.dirname(__file__), PREDEFINED_DIR, "backgrounds"
)

# Ensure directories exist
for p in [PREDEFINED_FOREGROUNDS_DIR, PREDEFINED_BACKGROUNDS_DIR]:
    # p.mkdir(parents=True, exist_ok=True)
    os.makedirs(p, exist_ok=True)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".JPG")

# Predefined images, now allowing both png and jpg
predefined_foregrounds = sorted(
    [f for f in os.listdir(PREDEFINED_FOREGROUNDS_DIR) if f.endswith(IMAGE_EXTENSIONS)]
)
predefined_backgrounds = sorted(
    [f for f in os.listdir(PREDEFINED_BACKGROUNDS_DIR) if f.endswith(IMAGE_EXTENSIONS)]
)


# --- Helper Functions ---


def run_extraction(input_path, output_path, 
                   filename, margin=20):
    """
    Wrapper to run the object extraction.
    This will create a 'foregrounds' directory inside output_path.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    extract_objects_from_labelme_data(input_path, output_path, margin, single_file_name=filename)


# Function to get the file path for a given filename


def get_foreground_path(filename):
    # return str(PREDEFINED_FOREGROUNDS_DIR / filename)
    return str(Path(PREDEFINED_FOREGROUNDS_DIR) / filename)


def get_background_path(filename):
    return str(Path(PREDEFINED_BACKGROUNDS_DIR) / filename)


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("Synthetic Data Generation Demo")

# Custom CSS for image selection
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


col1, col2 = st.columns(2)

# Initialize session state for selections
if "selected_foreground" not in st.session_state:
    st.session_state.selected_foreground = None
if "selected_background" not in st.session_state:
    st.session_state.selected_background = None
    
    
if "max_objects_per_image" not in st.session_state:
    st.session_state.max_objects_per_image = 3  # Default value

with col1:
    st.markdown("### Foreground Objects")
    st.write("Click an image to select it.")
    foreground_cols = st.columns(3)
    for i, fg_file in enumerate(predefined_foregrounds):
        with foreground_cols[i % 3]:
            # Use a button to make the image selectable
            is_selected = st.session_state.selected_foreground == fg_file
            container_style = (
                "selected-image-container" if is_selected else "image-container"
            )

            st.markdown(f'<div class="{container_style}">', unsafe_allow_html=True)
            if st.button(label=f"_{fg_file}_", key=f"fg_{i}", use_container_width=True):
                st.session_state.selected_foreground = fg_file
                st.rerun()  # Rerun to update the UI
            st.image(get_foreground_path(fg_file), caption=fg_file, width=150)
            st.markdown("</div>", unsafe_allow_html=True)


with col2:
    st.markdown("### Backgrounds")
    st.write("Click an image to select it.")
    background_cols = st.columns(3)
    for i, bg_file in enumerate(predefined_backgrounds):
        with background_cols[i % 3]:
            is_selected = st.session_state.selected_background == bg_file
            container_style = (
                "selected-image-container" if is_selected else "image-container"
            )

            st.markdown(f'<div class="{container_style}">', unsafe_allow_html=True)
            if st.button(label=f"_{bg_file}_", key=f"bg_{i}", use_container_width=True):
                st.session_state.selected_background = bg_file
                st.rerun()
            st.image(get_background_path(bg_file), caption=bg_file, width=150)
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# --- Blending Method Selection ---
st.markdown("### Blending Methoden auswählen")
col_blend1, col_blend2, col_blend3, col_blend4 = st.columns(4)
with col_blend1:
    use_alpha = st.checkbox("Alpha Blending", value=True)
with col_blend2:
    use_gauss = st.checkbox("Gaussian Blur", value=True)
with col_blend3:
    use_pyramid = st.checkbox("Pyramid Blending", value=False)
with col_blend4:
    use_poisson = st.checkbox("Poisson Blending", value=False)



selected_blending_methods = []
if use_alpha:
    selected_blending_methods.append(BlendingMode.ALPHA_BLENDING)
if use_gauss:
    selected_blending_methods.append(BlendingMode.GAUSSIAN_BLUR)
if use_pyramid:
    selected_blending_methods.append(BlendingMode.PYRAMID_BLEND)
if use_poisson:
    selected_blending_methods.append(BlendingMode.POISSON_BLENDING_MIXED)
    
# --- Max Objects Per Image Selection ---
st.markdown("### Max Objects Per Image")
st.slider(
    "Max Objects",
    min_value=1,
    max_value=10,
    value=st.session_state.max_objects_per_image,
    key="max_objects_per_image",
)

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

        st.info(f"Extracting object from {st.session_state.selected_foreground}...")
        # foreground_path = PREDEFINED_FOREGROUNDS_DIR / \
        #     st.session_state.selected_foreground
        foreground_path = os.path.join(
            PREDEFINED_FOREGROUNDS_DIR, st.session_state.selected_foreground
        )
        # Assuming the labelme JSON file has the same name as the image (e.g., image.png, image.json)
        # and is in the same directory.
        # run_extraction(str(foreground_path.parent),
        #                temp_extracted_dir, margin=20)
        run_extraction(str(PREDEFINED_FOREGROUNDS_DIR), temp_extracted_dir, 
                     filename=os.path.splitext(st.session_state.selected_foreground)[0],
                       margin=20)

        st.info("Generating synthetic image...")
        gen_input_dir = Path("gen_input")
        if gen_input_dir.exists():
            shutil.rmtree(gen_input_dir)
        os.makedirs(gen_input_dir)

        # Copy extracted objects and selected background into the input structure
        shutil.copytree(temp_extracted_dir, gen_input_dir / "foregrounds")
        os.makedirs(gen_input_dir / "backgrounds")
        shutil.copy(
            Path(PREDEFINED_BACKGROUNDS_DIR) / st.session_state.selected_background,
            gen_input_dir / "backgrounds",
        )

        # Run the generator
        data_generator = SyntheticImageGenerator(
            input_dir=str(gen_input_dir),
            output_dir=temp_output_dir,
            image_number=1,
            max_objects_per_image=st.session_state.max_objects_per_image,
            scaling_config=ScalingConfig(fixed_sizes=True),
            parallelize=False,
            blending_config=BlendingConfig(
                methods=selected_blending_methods,
                gaussian_kernel_size=9,
                gaussian_sigma=9,
                pyramid_blending_levels=1,
            )
        )
        data_generator.generate_images()

        # Display the result
        st.success("Image generated!")
        result_path = Path(temp_output_dir) / "images"
        print(f"Result path: {result_path}")
        image_list = list(result_path.glob("*"))
        print(f"Image list: {image_list}")
        if image_list:
            st.markdown("### Generated Image")
            img_cols = st.columns(2)
            for idx, img_path in enumerate(image_list):
                with img_cols[idx % 2]:
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)

        # Clean up
        shutil.rmtree(temp_extracted_dir)
        shutil.rmtree(temp_output_dir)
        shutil.rmtree(gen_input_dir)
    else:
        st.warning(
            "Please select both a foreground and a background image to generate."
        )
