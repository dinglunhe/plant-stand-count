import os
import gdown
import torch
import pandas as pd
from itertools import product
from ultralytics import YOLO 
from paalab.utils.utils import manage_cuda_memory

class YOLOHyperparameterTuner:
    def __init__(self, model_path="model/model.pt", data_path="data/data.yaml", output_dir="temp"):
        """
        Initialize the hyperparameter tuner.
        :param model_path: Path to the pre-trained YOLO model.
        :param data_path: Path to the dataset YAML file.
        :param output_dir: Directory to save results.
        """
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.results_summary = []
        self.best_map_per_freeze = {}
        self.best_params_per_freeze = {}

        os.makedirs(output_dir, exist_ok=True)

    def download_model(self, file_id="1n4tBmSAaBAa2nlYQ7eK6GSxf-9LXAYbY"):
        """
        Download the pre-trained model from Google Drive if not already present.
        :param file_id: Google Drive file ID.
        """
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            try:
                gdown.download(f"https://drive.google.com/uc?id={file_id}", self.model_path, quiet=False)
                print(f"Model downloaded successfully to {self.model_path}")
            except Exception as e:
                print(f"Error downloading model: {e}")
        else:
            print(f"Model already exists at {self.model_path}")

    def train_and_validate(self, epochs, patience, imgsz, params):
        """
        Train and validate the YOLO model with the given hyperparameters.
        :param epochs: Number of training epochs.
        :param patience: Early stopping patience.
        :param imgsz: Input image size.
        :param params: Dictionary of hyperparameters (lr, batch, weight_decay, dropout, freeze_layers).
        :return: Dictionary of results.
        """
        try:
            # Clear CUDA cache
            manage_cuda_memory('clear')

            # Load YOLO model
            if torch.cuda.is_available():
                model = YOLO(self.model_path).to("cuda:0")
            else:
                model = YOLO(self.model_path).to("cpu")

            assert os.path.exists(self.model_path), "Model file does not exist！"
            print(f"************Model loaded from {self.model_path}***********************************")
            # Train the model
            model.train(
                data=self.data_path,
                epochs=epochs,
                patience=patience,
                imgsz=imgsz,
                batch=params['batch'],
                lr0=params['lr'],
                lrf=params['lr'],
                weight_decay=params['weight_decay'],
                dropout=params['dropout'],
                freeze=params['freeze_layers'],
                cos_lr=True,
                verbose=True,
                pretrained=True,
                name=f"corn30_combine_freeze_{params['freeze_layers']}_lr_{params['lr']}_batch_{params['batch']}_wd_{params['weight_decay']}_dropout_{params['dropout']}",
                exist_ok=True,
            )

            # Validate the model
            results = model.val(
                data=self.data_path,
                imgsz=imgsz,
                batch=params['batch'],
                device="0",
            )

            current_map = results.results_dict.get("metrics/mAP50(B)", 0)
            print(f"Training completed: {params}, mAP50={current_map}")

            return {**params, "mAP50": current_map}

        except Exception as e:
            print(f"Error with parameters: {params}")
            print(f"Error message: {e}")
            return {**params, "mAP50": None}

    def run(self, hyperparams, epochs=30, patience=50, imgsz=640):
        """
        Perform hyperparameter tuning.
        :param hyperparams: Dictionary of hyperparameter ranges (e.g., lr, batch, freeze_layers).
        :param epochs: Number of training epochs.
        :param patience: Early stopping patience.
        :param imgsz: Input image size.
        """
        self.best_map_per_freeze = {freeze: 0 for freeze in hyperparams["freeze_layers"]}
        self.best_params_per_freeze = {freeze: {} for freeze in hyperparams["freeze_layers"]}

        # Generate all hyperparameter combinations
        param_combinations = product(
            hyperparams["lr"], hyperparams["batch"], hyperparams["weight_decay"], hyperparams["dropout"], hyperparams["freeze_layers"]
        )

        # Train and validate for each combination
        for lr, batch, weight_decay, dropout, freeze_layers in param_combinations:
            params = {
                "lr": lr,
                "batch": batch,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "freeze_layers": freeze_layers,
            }
            result = self.train_and_validate(epochs, patience, imgsz, params)

            # Update best results
            if result["mAP50"] and result["mAP50"] > self.best_map_per_freeze[freeze_layers]:
                self.best_map_per_freeze[freeze_layers] = result["mAP50"]
                self.best_params_per_freeze[freeze_layers] = result

            # Record results
            self.results_summary.append(result)

        # Save results to CSV
        self.save_results()

    def save_results(self):
        """
        Save all results and mark the best hyperparameter combinations.
        """
        df = pd.DataFrame(self.results_summary)
        df["is_best"] = df.apply(lambda row: row["mAP50"] == self.best_map_per_freeze.get(row["freeze_layers"], 0), axis=1)
        results_csv_path = os.path.join(self.output_dir, "hyperparameter_results.csv")
        df.to_csv(results_csv_path, index=False)

        print(f"\nResults saved to {results_csv_path}")
        print("\nBest hyperparameter combinations for each freeze layer:")
        for freeze, params in self.best_params_per_freeze.items():
            print(f"Freeze={freeze}: {params}")


# Main function to execute the tuner
if __name__ == "__main__":
    # Configuration
    tuner = YOLOHyperparameterTuner()

    # Download the model if not already present
    tuner.download_model("1n4tBmSAaBAa2nlYQ7eK6GSxf-9LXAYbY")

    # Hyperparameter ranges
    hyperparams = {
        "lr": [0.005, 0.01],
        "batch": [16, 32],
        "weight_decay": [0.0005, 0.001],
        "dropout": [0],
        "freeze_layers": [0, 7, 14]
        
    }

    ## Run hyperparameter tuning
    # tuner.run(hyperparams)
