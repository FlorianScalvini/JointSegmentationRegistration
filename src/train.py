"""
Training entry-point for the longitudinal brain MRI registration pipeline.

Parses command-line arguments, builds a timestamped output directory, wires
together the data module, the :class:`~pl_module.RegistrationLongitudinal`
Lightning module, and a :class:`~pytorch_lightning.Trainer`, then launches
training — optionally resuming from an existing checkpoint.

Loss weights (``lambda_*``), learning rate, precision, and all scheduling
parameters are fully configurable via CLI flags.

Usage::

    python train.py --dataset data/macaque.yaml --max_epochs 5000

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
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

# --- Local ---
from pl_module import PLJointRegistrationSegmentation
from dataloader.dataloader import SpatioTemporalSequenceDatamoduleJSON

def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns
    -------
    Namespace
        Parsed arguments with the following attributes:

        Paths
        ~~~~~
        dataset : str
            Path to the YAML dataset configuration file.
        checkpoint : str or None
            Path to a checkpoint to resume training from.

        Data
        ~~~~
        batch_size : int
            Training batch size.
        num_workers : int
            Number of DataLoader worker processes.
        size : int
            Spatial size (isotropic) of the model input volume.

        Training
        ~~~~~~~~
        max_epochs : int
            Maximum number of training epochs.
        pretrain_registration_epochs : int
            Number of initial epochs dedicated exclusively to registration.
        learning_rate : float
            Optimizer learning rate.
        lambda_seg : float
            Weight for the segmentation loss term.
        lambda_sim : float
            Weight for the image-similarity loss term.
        lambda_reg : float
            Weight for the regularisation loss term.
        lambda_jac : float
            Weight for the Jacobian-determinant loss term.
        precision : {16, 32}
            Floating-point precision used during training.
        num_sanity_val_steps : int
            Number of sanity validation steps before training starts.
        check_val_every_n_epoch : int
            Run validation every N epochs.
        checkpoint_every_n_steps : int
            Save a checkpoint every N training steps.
    """
    parser = argparse.ArgumentParser(
        description="Train the longitudinal brain MRI registration model."
    )

    # --- Paths ---
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/florian/VisualStudioProjects/JointSegmentationRegistration/data/babofet.yaml",
        help="Path to the dataset configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from (optional).",
    )
    parser.add_argument("--segmentation_checkpoint", type=str, default="/home/florian/VisualStudioProjects/JointSegmentationRegistration/last_segmentation_new.pt",
                        help="Optional pretrained segmentation state_dict.")
    parser.add_argument("--registration_checkpoint", type=str, default=None,
                        help="Optional pretrained registration state_dict.")

    # --- Data ---
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader worker processes.",
    )

    # --- Training ---
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=5000,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--pretrain_registration_epochs",
        type=int,
        default=400,
        help=(
            "Number of registration-only epochs before alternating "
            "registration and segmentation training."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--lambda_seg",
        type=float,
        default=1.0,
        help="Weight for the segmentation loss term.",
    )
    parser.add_argument(
        "--lambda_sim",
        type=float,
        default=0.0,
        help="Weight for the image-similarity loss term.",
    )
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=10.0,
        help="Weight for the regularisation loss term.",
    )
    parser.add_argument(
        "--lambda_jac",
        type=float,
        default=2.0,
        help="Weight for the Jacobian-determinant loss term.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        choices=[16, 32],
        help="Floating-point precision used during training.",
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        type=int,
        default=10,
        help="Number of sanity validation steps before training starts.",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=2,
        help="Run validation every N epochs.",
    )
    parser.add_argument(
        "--checkpoint_every_n_steps",
        type=int,
        default=10,
        help="Save a checkpoint every N training steps.",
    )
    parser.add_argument(
        "--lambda_anchor",
        type=float,
        default=1.0,
        help="Weight for supervised pretraining-data loss that prevents segmentation collapse.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.99,
        help="EMA decay used by the teacher that generates temporal pseudo-labels.",
    )
    parser.add_argument(
        "--segmentation_method",
        choices=["temporal_neighbors", "t0_anchor"],
        default="temporal_neighbors",
        help=(
            "Segmentation co-training strategy: EMA temporal neighbours or "
            "propagation of the ground-truth segmentation at t0."
        ),
    )

    parser.add_argument(
        "--weight",
        action="store_true",
        help="Weight the registration loss",
    )

    return parser.parse_args()


def main(args: Namespace) -> None:
    """Build all components and launch training.

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
    save_dir: str = os.path.join("./", "results", config["name"], "train", dir_name)
    if os.path.exists(save_dir):
        # create versioned directory if the base directory already exists
        version: int = 1
        while os.path.exists(f"{save_dir}_v{version}"):
            version += 1
        save_dir = f"{save_dir}_v{version}"
    os.makedirs(save_dir, exist_ok=True)

    # --- Logger ---
    tensorboard_logger: TensorBoardLogger = TensorBoardLogger(
        save_dir=save_dir,
        name="tensorboard",
        version=0,
        default_hp_metric=False,
    )

    # --- Data module ---
    # The pretraining database supplies random labeled anchor samples, while
    # the sequence database supplies the longitudinal subject being optimized.
    # Keep the old key names as fallbacks for existing dataset YAML files.
    pretrain_json = config.get("pretrain_json", config.get("train_json"))
    sequence_json = config.get("sequence_json", config.get("val_json"))
    if pretrain_json is None or sequence_json is None:
        raise KeyError(
            "dataset YAML must define pretrain_json and sequence_json "
            "(or legacy train_json and val_json)"
        )
    datamodule: pl.LightningDataModule = SpatioTemporalSequenceDatamoduleJSON(
        root_dir=config["root_dir"],
        json_path=pretrain_json,
        json_path_val=sequence_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        t0=config["t0"],
        tn=config["tn"],
    )

    # --- Model ---
    training_module: PLJointRegistrationSegmentation = PLJointRegistrationSegmentation(
        learning_rate=args.learning_rate,
        num_classes=config["num_classes"],
        save_dir=save_dir,
        lambda_seg=args.lambda_seg,
        lambda_reg=args.lambda_reg,
        lambda_sim=args.lambda_sim,
        lambda_jac=args.lambda_jac,
        t0=config["t0"],
        tn=config["tn"],
        lambda_anchor=args.lambda_anchor,
        ema_decay=args.ema_decay,
        pretrain_registration_epochs=args.pretrain_registration_epochs,
        segmentation_method=args.segmentation_method,
        weight=args.weight,
        shape=config["rsize"],
        step_time=0.1,
    )

    # --- Trainer ---
    trainer: pl.Trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        num_sanity_val_steps=args.num_sanity_val_steps,
        logger=tensorboard_logger,
        callbacks=[
            ModelCheckpoint(
                every_n_train_steps=args.checkpoint_every_n_steps,
                dirpath=save_dir,
                save_last=True,
            ),
            TQDMProgressBar(refresh_rate=1),
        ],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        enable_progress_bar=True,
    )
    if args.segmentation_checkpoint:
        training_module.segmentation.load_state_dict(
            torch.load(args.segmentation_checkpoint, map_location="cpu")
        )
        training_module.sync_segmentation_ema()
    if args.registration_checkpoint:
        training_module.registration.load_state_dict(
            torch.load(args.registration_checkpoint, map_location="cpu")
        )
    trainer.fit(model=training_module, datamodule=datamodule, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main(args=parse_args())
