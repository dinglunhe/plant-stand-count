import os
import pandas as pd
import torch
from ultralytics import YOLO 
from paalab.utils.utils import manage_cuda_memory

class ModelSelector:
    def __init__(self, results_csv_path="temp/hyperparameter_results.csv", base_model_path="runs/detect", data_yaml="data/data.yaml"):
        """
        Initialize the model selector with paths.

        :param results_csv_path: Path to the CSV file with training results.
        :param base_model_path: Base directory where trained models are stored.
        :param data_yaml: Path to the dataset configuration YAML file.
        """
        self.results_csv_path = results_csv_path
        self.base_model_path = base_model_path
        self.data_yaml = data_yaml
        self.best_model_path = None
        self.best_params = None

    def load_results(self):
        """Load and validate the results CSV file."""
        if not os.path.exists(self.results_csv_path):
            print(f"Results file does not exist: {self.results_csv_path}")
            return None

        df = pd.read_csv(self.results_csv_path).dropna(subset=['mAP50'])

        if df.empty:
            print("No valid mAP50 results available for selection.")
            return None

        return df

    def select_best_model(self):
        """Select the best model based on mAP50 score."""
        df = self.load_results()
        if df is None:
            return None

        best_row = df.loc[df['mAP50'].idxmax()]
        self.best_params = {
            'freeze_layers': best_row['freeze_layers'],
            'lr': best_row['lr'],
            'batch': best_row['batch'],
            'weight_decay': best_row['weight_decay'],
            'dropout': best_row['dropout'],
            'mAP50': best_row['mAP50']
        }

        model_dir_name = f"corn30_combine_freeze_{best_row['freeze_layers']}_lr_{best_row['lr']}_batch_{best_row['batch']}_wd_{best_row['weight_decay']}_dropout_{best_row['dropout']}"
        self.best_model_path = os.path.join(self.base_model_path, model_dir_name, "weights", "best.pt")
        print(self.best_model_path)
        if not os.path.exists(self.best_model_path):
            print(f"The best model file does not exist: {self.best_model_path}")
            return None

        return self.best_model_path

    def validate_best_model(self, batch_size=4):
        """Load and validate the best model."""
        if self.best_model_path is None:
            print("No valid model found for validation.")
            return None

        manage_cuda_memory("clear") # Clear CUDA cache
        model = YOLO(self.best_model_path)

        print("\nBest model parameters:")
        for key, value in self.best_params.items():
            print(f"{key}: {value}")

        if torch.cuda.is_available():
            results = model.val(data=self.data_yaml, imgsz=640, batch=batch_size, device="0", )
        else:
            results = model.val(data=self.data_yaml, imgsz=640, batch=batch_size, device="cpu", )

        # Optional: Uncomment to display results
        # print("\nValidation results:")
        # print(results)

        return results

if __name__ == "__main__":
    # Run model selection and validation
    selector = ModelSelector()
    if selector.select_best_model():
        selector.validate_best_model()
