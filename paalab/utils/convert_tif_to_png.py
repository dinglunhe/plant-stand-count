import os
import rasterio
from PIL import Image
import numpy as np

def create_output_folder(output_folder):
    """Create output folder if it does not exist."""
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder '{output_folder}' created or already exists.")

def determine_band_order(descriptions):
    """Determine band order based on band descriptions."""
    if descriptions == ('Blue', 'Green', 'Red'):
        print("Detected band order as BGR.")
        return 'BGR'
    print("Detected band order as RGB.")
    return 'RGB'

def process_image(src, band_order):
    """Process the image and return an RGB image."""
    bands = src.read()
    
    if bands.shape[0] < 3:
        raise ValueError("The file does not contain enough bands for RGB conversion.")
    
    if band_order == 'BGR':
        rgb_image = np.dstack((bands[2], bands[1], bands[0]))  # BGR -> RGB
    else:
        rgb_image = np.dstack((bands[0], bands[1], bands[2]))  # Default RGB
    
    # Normalize to 0-255 range and convert to 8-bit unsigned integer
    rgb_image = ((rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min()) * 255).astype(np.uint8)
    return rgb_image

def save_image(rgb_image, output_folder, filename):
    """Save the RGB image as a PNG file."""
    output_path = os.path.join(output_folder, filename.replace('.tif', '.png'))
    img = Image.fromarray(rgb_image)
    img.save(output_path)
    print(f"File '{filename}' successfully converted to '{output_path}'.")

def convert_tif_to_png(image_folder, output_folder):
    """Convert TIFF files in a folder to PNG files."""
    print(f"Starting to process TIFF files in folder '{image_folder}'.")
    create_output_folder(output_folder)
    
    band_order = None
    
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".tif"):
            try:
                with rasterio.open(os.path.join(image_folder, filename)) as src:
                    if band_order is None:
                        descriptions = src.descriptions
                        print(f"Band descriptions for file '{filename}': {descriptions}")
                        band_order = determine_band_order(descriptions)
                    
                    rgb_image = process_image(src, band_order)
                    save_image(rgb_image, output_folder, filename)
            
            except Exception as e:
                print(f"Error processing file '{filename}': {e}")

# Test function
def test_conversion():
    """Test the TIFF-to-PNG conversion functionality."""
    test_image_folder = "test_images"
    test_output_folder = "test_output"
    
    # Create test folders and test file
    os.makedirs(test_image_folder, exist_ok=True)
    os.makedirs(test_output_folder, exist_ok=True)
    
    # Generate a simple test TIFF file
    test_tiff_path = os.path.join(test_image_folder, "test_image.tif")
    test_data = np.random.randint(0, 256, (3, 100, 100), dtype=np.uint8)  # Randomly generate a 3-band image
    with rasterio.open(
        test_tiff_path,
        'w',
        driver='GTiff',
        height=100,
        width=100,
        count=3,
        dtype=np.uint8
    ) as dst:
        dst.write(test_data[0], 1)
        dst.write(test_data[1], 2)
        dst.write(test_data[2], 3)
    
    print("Test TIFF file created.")
    
    # Run the conversion function
    convert_tif_to_png(test_image_folder, test_output_folder)
    
    # Check if the output file exists
    output_file = os.path.join(test_output_folder, "test_image.png")
    if os.path.exists(output_file):
        print("Test passed: Output PNG file generated.")
    else:
        print("Test failed: Output PNG file not generated.")

# Main program
if __name__ == "__main__":
    # Run the test
    test_conversion()
    
    ## Actual conversion
    # image_folder = 'test_pipeline_tif_data'
    # output_folder = 'test_pipeline_png_output'
    # convert_tif_to_png(image_folder, output_folder)