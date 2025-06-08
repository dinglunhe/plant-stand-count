import torch
from ultralytics import YOLO
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import rasterio
import glob


def check_cuda_status():
    # Check whether CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs available: {gpu_count}")
        
        # Iterate over each GPU and print detailed info
        for i in range(gpu_count):
            print(f"\n--- GPU {i} Info ---")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
            print(f"Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**2:.2f} MB")
        
        # Current device
        current_device = torch.cuda.current_device()
        print(f"\nCurrent Device ID: {current_device}")
        print(f"Current Device Name: {torch.cuda.get_device_name(current_device)}")
        
        # Check CUDA driver and library version
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        print(f"\nCUDA Version: {cuda_version}")
        print(f"cuDNN Version: {cudnn_version}")
        
        # Check if cuDNN is enabled
        cudnn_enabled = torch.backends.cudnn.enabled
        print(f"cuDNN Enabled: {cudnn_enabled}")
    else:
        print("No GPUs available.")

def manage_cuda_memory(action="check"):
    """
    A utility function to manage CUDA memory.
    
    Parameters:
        action (str): Action to perform. Options are:
            - "check": Print current CUDA memory usage.
            - "clear": Clear unused GPU memory and print the status.

    Returns:
        None
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. No action performed.")
        return
    
    if action == "check":
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    elif action == "clear":
        print("Clearing CUDA memory...")
        torch.cuda.empty_cache()
        print("After clearing:")
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    else:
        print(f"Unknown action: {action}. Please use 'check' or 'clear'.")


def get_best_f1_and_threshold(curves_results):
    """
    Get the best F1 score and the corresponding confidence threshold.

    Parameters:
        curves_results (list): A list where `curves_results[1][0]` contains confidence thresholds 
                                and `curves_results[1][1]` contains F1 scores.

    Returns:
        tuple: (best_f1, best_confidence) where
                - best_f1 (float): The maximum F1 score.
                - best_confidence (float): The confidence threshold corresponding to the best F1 score.
    """
    # Extract confidence thresholds and F1 scores
    confidence_thresholds = np.array(curves_results[1][0])
    f1_scores = np.array(curves_results[1][1])

    # Find the maximum F1 score and its index
    best_f1 = np.max(f1_scores)
    best_idx = np.argmax(f1_scores)

    # Get the corresponding confidence threshold
    best_confidence = confidence_thresholds[best_idx]

    return best_f1, best_confidence

def inference(model_path, image_path, conf_threshold=0.5, iou_threshold=0.45, output_dir="./temp"):
    """
    Perform inference using a YOLO model on the specified image or directory and save results.

    Parameters:
        model_path (str): Path to the trained YOLO model (.pt file).
        image_path (str): Path to the image or directory for inference.
        conf_threshold (float): Confidence threshold for filtering predictions. Default is 0.5.
        iou_threshold (float): IoU threshold for Non-Max Suppression. Default is 0.45.
        output_dir (str): Directory to save results. If None, uses the default YOLO output path.

    Returns:
        pred_results: Prediction results from the YOLO model.
    """
    # Check if GPU is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # Load the model
    model = YOLO(model_path)

    # Perform inference
    pred_results = model.predict(
        source=image_path,
        imgsz=640,
        conf=conf_threshold,
        iou=iou_threshold,
        project=output_dir,
        save=True,
        save_txt=True,
        save_conf=False,
        device=device,
        visualize=False,
        show=False
    )

    return pred_results

def save_predictions_to_csv(pred_results, output_dir="./results", csv_filename="predictions.csv"):
    """
    Extract bounding box coordinates from YOLO predictions and save them to a CSV file.

    Parameters:
        pred_results (list): List of YOLO prediction results.
        output_dir (str): Directory to save the CSV file. Default is "./results".
        csv_filename (str): Name of the output CSV file. Default is "predictions.csv".

    Returns:
        csv_output_path (str): Path of the saved CSV file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the path to save the CSV file
    csv_output_path = os.path.join(output_dir, csv_filename)

    # Initialize an empty list to store prediction data
    pred_data = []

    # Iterate through prediction results
    for result in pred_results:
        image_name = os.path.basename(result.path)  # Extract image name

        # Iterate through detected bounding boxes
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            confidence = box.conf[0].item()  # Get confidence score
            class_id = int(box.cls[0].item())  # Get class ID

            # Append data to the list
            pred_data.append([image_name, class_id, x1, y1, x2, y2, confidence])

    # Convert the list to a DataFrame
    df = pd.DataFrame(pred_data, columns=["image_name", "class_id", "x1", "y1", "x2", "y2", "confidence"])

    # Save DataFrame to CSV
    df.to_csv(csv_output_path, index=False)

    print(f"✅ Predictions saved to: {csv_output_path}")

    return csv_output_path

def show_predictions(pred_results, output_dir="./results/show_predictions", display=True):
    """
    Process and save images with detected bounding boxes.

    Parameters:
        pred_results (list): List of YOLO prediction results.
        output_dir (str): Directory to save processed images. Default is "./results/show_predictions".
        display (bool): Whether to display images after processing. Default is True.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for result in pred_results:
        # Get a copy of the original image to avoid modifying the original
        img = result.orig_img.copy()

        # Iterate through all detected bounding boxes
        for box in result.boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw the bounding box on the image (Green color, thickness = 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Extract the file name from the original image path
        image_name = os.path.basename(result.path)

        # Define the save path for the processed image
        save_path = os.path.join(output_dir, image_name)

        # Save the image (Ensure it remains in BGR format for OpenCV)
        cv2.imwrite(save_path, img)

        print(f"✅ Processed image saved to: {save_path}")

        # Display the image if enabled
        if display:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct visualization
            plt.imshow(img_rgb)
            plt.axis('off')  # Hide the axes
            plt.show()

def convert_csv_to_geo(csv_file, tif_dir, output_dir):
    """
    Checks if CSV and TIFF match and converts pixel coordinates to geographic coordinates.
    
    Parameters:
        csv_file (str) : path to the CSV file for target detection.
        tif_dir (str): the folder where the TIFF images are stored.
        output_dir (str): the folder where the results are stored
        
    Result:
        - Generate `detection_results_geo.csv` with geographic coordinates `geo_x, geo_y`
        - Check if TIFF image matches CSV
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all TIFF files
    list_tif = glob.glob(os.path.join(tif_dir, "*.tif"))
    tif_map = {os.path.basename(tif).replace(".tif", ""): tif for tif in list_tif}  # {test_1: path}

    # Read CSV data
    df = pd.read_csv(csv_file)

    # Collect all image names in CSV (remove .png suffix)
    csv_images = set(df["image_name"].str.replace(".png", "", regex=False))

    # Check whether TIFF and CSV match
    tif_set = set(tif_map.keys())
    missing_tifs = csv_images - tif_set  # Exists in CSV but not in TIFF
    extra_tifs = tif_set - csv_images    # Exists in TIFF but not in CSV

    if missing_tifs:
        print(f"❌ Missing corresponding TIFF image: {missing_tifs}")
    if extra_tifs:
        print(f"⚠️ TIFF image exists without a CSV record: {extra_tifs}")

    # Iterate through CSV and convert by image
    geo_results = []
    for img_name in csv_images:
        if img_name in tif_map:
            with rasterio.open(tif_map[img_name]) as dataset:
                # Extract detection box data for the current image
                img_data = df[df["image_name"] == f"{img_name}.png"]

                def pixel2coord(x, y):
                    """Pixel coordinates (x, y) → Geographic coordinates (lon, lat)"""
                    lon, lat = dataset.xy(y, x)
                    return lon, lat

                # Compute center point and convert to geographic coordinates
                geo_x, geo_y = [], []
                for i in range(len(img_data)):
                    cx = (img_data["x1"].iloc[i] + img_data["x2"].iloc[i]) / 2
                    cy = (img_data["y1"].iloc[i] + img_data["y2"].iloc[i]) / 2
                    lon, lat = pixel2coord(cx, cy)
                    geo_x.append(lon)
                    geo_y.append(lat)

                # Add geographic coordinates to DataFrame
                img_data = img_data.copy()
                img_data["geo_x"] = geo_x
                img_data["geo_y"] = geo_y
                geo_results.append(img_data)

    # Combine all results and save
    if geo_results:
        geo_df = pd.concat(geo_results, ignore_index=True)
        geo_csv_path = os.path.join(output_dir, "detection_results_geo.csv")
        geo_df.to_csv(geo_csv_path, index=False)
        print(f"✅ Processing completed: {geo_csv_path}")
        print(geo_df.head())
    else:
        print("⚠️ No convertible data")
