"""
Test entry-point for the longitudinal brain MRI registration pipeline.

Loads a trained :class:`RegistrationLongitudinal` model from a checkpoint,
runs inference on the test split defined by the dataset YAML, and then
evaluates the predictions by computing segmentation Dice scores, the number
of non-positive Jacobian determinants (deformation regularity), gyrification
index (GI), and optional VTK / image-slice exports.

Author : Florian Scalvini
"""

# --- Standard library ---
import gc
import os
import argparse
from argparse import Namespace
from datetime import datetime
from typing import Any, Dict

# --- Third-party ---
import torch
import yaml
import pytorch_lightning as pl

# --- Local ---
import utils.compute_evaluation as evaluation
from model.neural_ode import RegistrationLongitudinal
from dataloader.dataloader import SpatioTemporalSequenceDatamoduleJSON




def main(args: Namespace) -> None:
    """Build all components and launch testing.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments returned by :func:`parse_args`.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")

    with open(args.dataset, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # --- Output directory ---
    dir_name: str = datetime.now().strftime("%y_%d_%H_%M")
    save_dir: str = os.path.join("./", "results", config["name"], "test", dir_name)
    if os.path.exists(save_dir):
        # create versioned directory if the base directory already exists
        version: int = 1
        while os.path.exists(f"{save_dir}_v{version}"):
            version += 1
        save_dir = f"{save_dir}_v{version}"
    os.makedirs(save_dir, exist_ok=True)

    # --- Data module ---
    datamodule: pl.LightningDataModule = SpatioTemporalSequenceDatamoduleJSON(
        root_dir=config["root_dir"],
        json_path=config["train_json"],
        json_path_val=config["val_json"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        size=config["csize"],
        crop=config["rsize"],
        t0=config["t0"],
        tn=config["tn"],
    )

    # --- Model ---
    training_module: RegistrationLongitudinal = RegistrationLongitudinal(
        learning_rate=args.learning_rate,
        save_dir=save_dir,
        shape=args.size,
        step_time=0.1
    )

    # --- Trainer ---
    trainer: pl.Trainer = pl.Trainer(
        precision=32,
        enable_progress_bar=True,
    )

    training_module.model.load_state_dict(torch.load(args.weight_path, weights_only=True))
    
    # Generate predictions 
    trainer.test(model=training_module, datamodule=datamodule)
    # Evaluate predictions based on the evaluation metrics (e.g. the Dice score for segmentation, the number of non-positive Jacobian determinants for deformation regularity, etc.) and save the results in the output directory.

    # Duplicate 
    evaluation.compute_evaluation(dataset_yaml=args.dataset_yaml, 
                                  pred_path=os.path.join(save_dir, "predictions"),
                                  compute_dice=True, 
                                  create_vtk=True, 
                                  compute_flow=True, 
                                  compute_gi=True, 
                                  create_img_slice=False, 
                                  plane_idx=0, 
                                  gi_normalized_csv=config['gi_normalized'], 
                                  rotate=args.rotate)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the longitudinal brain MRI registration model."
    )
    # --- Paths ---
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/florian/PyCharmMiscProject/data/macaque.yaml",
        help="Path to the dataset configuration file.",
    )
    
    parser.add_argument(
        "--weight_path",
        type=str,
        default="/home/florian/PyCharmMiscProject/results/dhcpatlas/train/26_07_15_55_v1/best_registration.pt",
        help="Path to the weight file for testing.",
    )
  
    args = parser.parse_args()
    main(args=args)
