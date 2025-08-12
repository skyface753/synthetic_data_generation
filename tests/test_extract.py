import pytest
from syndatagenyolo.extract import extract_objects_from_labelme_data
import os

# files are in data directory
@pytest.mark.parametrize(
    "input_dir, output_dir, margin, single_file_name",
    [
        ("tests/data/foregrounds_plain", "tests/data/tmp/extracted_objects", 10, None),
        ("tests/data/foregrounds_plain", "tests/data/tmp/extracted_objects_single", 10, "green_ball"),
    ],
)
def test_extract_objects_from_labelme_data(input_dir, output_dir, margin, single_file_name):
    """
    Test the extraction of objects from LabelMe data.
    """
    # Call the function to extract objects
    extract_objects_from_labelme_data(input_dir, output_dir, margin, single_file_name)

    # Check if the output directory exists
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist"

    # Check if images are created in the output directory
    extracted_images = os.listdir(output_dir)
    assert len(extracted_images) > 0, "No images were extracted"
    
    # If single_file_name is provided, check for a specific file
    if single_file_name:
        extracted_images = os.listdir(output_dir + f"/{single_file_name}")
        assert f"0-{single_file_name}.png" in extracted_images or f"0-{single_file_name}.jpg" in extracted_images, \
            f"Expected file 0-{single_file_name} not found in {output_dir}"

    
