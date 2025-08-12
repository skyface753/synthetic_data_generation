import pytest
from syndatagenyolo.data_generation import SyntheticImageGenerator, BlendingConfig, BlendingMode
from pathlib import Path

def test_synthetic_image_generator_initialization():
    """
    Test the initialization of the SyntheticImageGenerator class.
    """
    input_dir = "tests/data/syn_data_gen_input"
    output_dir = "tests/data/tmp/syn_data_gen_output"
    image_number = 10
    image_width = 640
    image_height = 480
    methods = [BlendingMode.ALPHA_BLENDING, BlendingMode.GAUSSIAN_BLUR, BlendingMode.POISSON_BLENDING_MIXED, BlendingMode.POISSON_BLENDING_NORMAL, BlendingMode.PYRAMID_BLEND]
    # Initialize the generator
    generator = SyntheticImageGenerator(
        input_dir=input_dir,
        output_dir=output_dir,
        image_number=image_number,
        image_width=image_width,
        image_height=image_height,
        blending_config=BlendingConfig(
            methods= methods,
            gaussian_kernel_size = 15,
            gaussian_sigma = 30,
            pyramid_blending_levels=3
            
        ),
        overwrite_output=True
    )

    # Check if the generator is initialized correctly
    print(f"Input Directory: {generator.input_dir}")
    assert generator.input_dir == Path(input_dir), "Input directory not set correctly"
    assert generator.output_dir == Path(output_dir), "Output directory not set correctly"
    assert generator.image_number == image_number, "Image number not set correctly"
    assert generator.image_width == image_width, "Image width not set correctly"
    assert generator.image_height == image_height, "Image height not set correctly"
    
    generator.generate_images()  # Call the method to generate images
    assert Path(output_dir).exists(), "Output directory was not created"
    num_gen_images = len(list(Path(output_dir + "/images").glob("*.jpg")))  # Assuming images are saved as .jpg
    expected_num_images = image_number * len(methods)  # Each method generates images
    assert num_gen_images == expected_num_images, f"Expected {expected_num_images} images, but found {num_gen_images}"
    
    # check if the generated classes.txt file has the same classes as the provided classes.yaml
    class_map_path = "classes.yaml"
    from syndatagenyolo.utils.classes_yaml import load_class_map
    classes_map = load_class_map(class_map_path)
    assert classes_map is not None, "Class map should not be None"
    
    # check if the classes.txt file exists in the output directory
    classes_txt_path = Path(output_dir) / "classes.txt"
    assert classes_txt_path.exists(), f"classes.txt file does not exist in {output_dir}"
    with open(classes_txt_path, 'r') as f:
        classes_txt_content = f.read().strip().splitlines()
        assert len(classes_txt_content) == len(classes_map), "Number of classes in classes.txt does not match the class map"
        for class_name in classes_map.keys():
            assert class_name in classes_txt_content, f"Class '{class_name}' not found in classes.txt"
        print(f"Classes in classes.txt: {classes_txt_content}")
        print(f"Classes in class map: {list(classes_map.keys())}")
        
    
if __name__ == "__main__":
    test_synthetic_image_generator_initialization()