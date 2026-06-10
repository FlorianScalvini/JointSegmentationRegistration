# --- Standard library ---
import os
import json
import random

# --- Third-party ---
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import pytorch_lightning as pl
import torchio as tio
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
from pytorch_lightning.utilities.types import STEP_OUTPUT

# --- Local ---
import utils.utils as utils
import utils.losses as losses
import utils.visualize as visualize
import utils.registration as registration
from model.neural_ode import LongitudinalODERegistration


class RegistrationLongitudinal(pl.LightningModule):
    """PyTorch Lightning module for longitudinal brain image registration using Neural ODEs.

    Integrates a deformation-field ODE model with a multi-term loss (similarity,
    segmentation, regularisation, SDF, Jacobian determinant penalty) and logs
    training / validation metrics to TensorBoard.
    """

    # ──────────────────────────────────────────────────────────────────────────
    #  Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        learning_rate: float = 0.01,
        save_dir: str = "",
        lambda_seg: float = 1,
        lambda_reg: float = 0.001,
        lambda_sdf: float = 1,
        lambda_sim: float = 0.0,
        lambda_jac: float = 0.000001,
        shape: list[int] = [192, 224, 192],
        step_time: float = 0.1,
        *args,
        **kwargs,
    ) -> None:
        """Initialise model, loss functions, metrics, and tracking variables."""
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.learning_rate = learning_rate
        # Initialize the registration and segmentation networks
        self.model = LongitudinalODERegistration(shape=shape, step_time=step_time)

        # Hyperparameters
        self.lambda_sdf = lambda_sdf
        self.lambda_reg = lambda_reg
        self.lambda_sim = lambda_sim
        self.lambda_seg = lambda_seg
        self.lambda_jac = lambda_jac

        # Loss functions and metrics
        self.loss_sim = monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=21) # type: ignore
        self.loss_reg = losses.Grad3d('l2')
        self.loss_sdf = nn.L1Loss()
        self.loss_seg = nn.MSELoss()
        self.loss_jac = losses.NonDetJacobianPenalty()

        self.seg_metrics = monai.metrics.DiceMetric() # type: ignore

        # Logging and tracking best performance
        self.save_dir = save_dir
        self.max_dice_score = 0
        self.table_result_data = []
        self.val_grid_images = []

    # ──────────────────────────────────────────────────────────────────────────
    #  Forward pass
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        ages: torch.Tensor,
        target_age: torch.Tensor,
        grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the ODE registration and rescale deformation fields to voxel space."""
        shape = source.shape[2:]
        scale_factor = torch.tensor(shape).to(self.device).view(1, 3, 1, 1, 1) * 1.
        all_phi, loss_reg = self.model(source, target, ages, target_age, grid)
        all_phi = (all_phi + 1.) / 2. * scale_factor
        return all_phi, loss_reg

    # ──────────────────────────────────────────────────────────────────────────
    #  Training
    # ──────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self) -> tuple[list, list]:
        """Return Adam optimiser with exponential LR decay."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: tuple, batch_idx: int) -> None:
        """Compute total weighted loss, back-propagate, and log per-term metrics."""
        optimizer = self.optimizers()

        images, segs, ages = batch
        shape = images[0].shape[2:]
        scale_factor = torch.tensor(shape).to(self.device).view(1, 3, 1, 1, 1) * 1.
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)

        images = images.squeeze(0)
        segs = segs.squeeze(0)
        ages = ages.squeeze(0).to(self.device)


        for idx_i in range(0, images.shape[0]-1):
            for idx_j in range(idx_i + 1, images.shape[0]):
                target_img = images[idx_j:idx_j + 1].float()
                initial_img = images[idx_i:idx_i + 1].float()
                initial_seg = F.one_hot(segs[idx_i:idx_i + 1].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3)
                all_phi, loss_reg = self(initial_img, target_img, ages[idx_i:], ages[idx_j], grid)
                grid_voxel = (grid + 1.) / 2. * scale_factor
                loss_sim = torch.tensor(0.0, device=self.device)
                loss_seg = torch.tensor(0.0, device=self.device)
                loss_jac = torch.tensor(0.0, device=self.device)

                for idx in range(1, all_phi.shape[0]):
                    phi = all_phi[idx]
                    df = phi - grid_voxel
                    if self.lambda_sim > 0:
                        warped = registration.warp(initial_img, df)
                        loss_sim += self.loss_sim(warped, images[idx+idx_i:idx + idx_i + 1].float())
                        del warped
                    if self.lambda_seg > 0:
                        warped_seg = registration.warp(initial_seg.float().to(self.device), df)
                        loss_seg += self.loss_seg(warped_seg, F.one_hot(segs[idx+idx_i:idx + idx_i + 1].squeeze(0).cpu().long(), num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3).float().to(self.device))
                        del warped_seg
                    if self.lambda_jac > 0:
                        loss_jac += self.loss_jac(df)
                    del phi, df

                num_steps = images.shape[0] - 1
                loss_seg = loss_seg / num_steps
                loss_sim = loss_sim / num_steps
                loss_jac = loss_jac / num_steps
                loss_reg = loss_reg / ((ages[-1] - ages[0]) / self.model.step_time) # Normalize by number of integration steps, not number of images
                loss =  self.lambda_sim * loss_sim + self.lambda_seg * loss_seg  + self.lambda_reg * loss_reg + self.lambda_jac * loss_jac
                optimizer.zero_grad() # type: ignore
                self.manual_backward(loss)
                optimizer.step() # type: ignore

                self.log_dict({
                    'loss_G': loss.item(),
                    'loss_sim': (self.lambda_sim * loss_sim).item(),
                    'loss_seg': (self.lambda_seg * loss_seg).item(),
                    'loss_reg': (self.lambda_reg * loss_reg).item(),
                    'loss_jac': (self.lambda_jac * loss_jac).item()
                }, on_step=False, on_epoch=True, prog_bar=True)

                # ── critical: free the ODE trajectory ──
                del all_phi, grid_voxel, loss, loss_sim, loss_reg
                # ── always flush at end of step ──
                torch.cuda.empty_cache()

    def on_train_epoch_end(self) -> None:
        """Flush GPU cache and save a checkpoint at the end of each training epoch."""
        torch.cuda.empty_cache()  # ← add this
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "last_registration.pt"))

    # ──────────────────────────────────────────────────────────────────────────
    #  Validation
    # ──────────────────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self) -> None:
        """Reset per-epoch validation accumulators before the validation loop."""
        self.val_grid_images = []
        self.table_result_data = []

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Register images, compute Dice / Jacobian metrics, and collect visualisations."""
        # Initialization images
        all_registered = []
        all_targets = []
        all_segs = []

        images, segs, ages = batch
        shape = images[0].shape[2:]
        scale_factor = torch.tensor(shape).to(self.device).view(1, 3, 1, 1, 1) * 1.
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)
        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        shape = images.shape[2:]
        initial_img = images[0:1].float()
        target_img = images[-1:].float()
        with torch.no_grad():
            all_phi, _ = self(initial_img, target_img, ages, ages[-1], grid)
        all_phi = all_phi.detach()
        grid_voxel = (grid + 1.) / 2. * scale_factor
        all_registered = []
        all_targets = []
        all_segs = []

        initial_seg = F.one_hot(segs[:, 0].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3)
        for idx in range(0, images.shape[0]):
            phi = all_phi[idx]
            df = phi - grid_voxel
            warped = registration.warp(images[0:1].float(), df)
            warped_seg = registration.warp(initial_seg.to(self.device).float(), df)
            warped_seg = torch.argmax(warped_seg, dim=1).detach()
            pred_label = F.one_hot(warped_seg.cpu().long(), num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3)

            all_registered.append(
                utils.normalize_to_0_1(warped.squeeze())[:, :, shape[-1] // 2].detach().cpu().unsqueeze(0).repeat(3, 1, 1)
            )
            all_targets.append(
                utils.normalize_to_0_1(images[idx].squeeze(0))[:, :, shape[-1] // 2].detach().cpu().unsqueeze(0).repeat(3, 1,
                                                                                                                  1)
            )
            all_segs.append(
                utils.normalize_to_0_1(warped_seg.squeeze())[:, :, shape[-1] // 2].detach().cpu().unsqueeze(0).repeat(3, 1, 1)
            )
            xy = registration.displacement2grid(df.cpu()).squeeze(0).detach()
            grid_img = visualize.plt_grid(xy[:, :, shape[-1] // 2, :].cpu())[0]
            to_tensor = transforms.ToTensor()
            grid_img = to_tensor(grid_img)  # (3, H, W)

            if idx != 0:
                self.seg_metrics(pred_label, F.one_hot(segs[:, idx].squeeze(0).cpu().long(),
                                                       num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3).cpu())
                det_jac = utils.compute_jacobian_determinant_3d(df.cpu()).numpy()
                nb_jac_neg = int(np.sum(det_jac < 0))
                buffer = self.seg_metrics.get_buffer()
                dice = float(buffer[-1].mean().item())
                results = [str(batch_idx) + "_" + str(idx), grid_img, dice, nb_jac_neg]
                self.table_result_data.append(results)

            del warped, warped_seg, phi, xy, pred_label
            torch.cuda.empty_cache()

        del all_phi, df
        torch.cuda.empty_cache()

        num_times = images.shape[0]
        combined = torch.stack(all_targets + all_registered + all_segs)
        grid_visualization = make_grid(combined, nrow=num_times, padding=5, pad_value=1.0)
        self.val_grid_images.append(grid_visualization)
        del combined

    def on_validation_epoch_end(self) -> None:
        """Log aggregated metrics and grid images; save model if a new Dice best is reached."""
        if not self.table_result_data:  # skip sanity check
            self.seg_metrics.reset()
            return

        step = self.current_epoch

        # Log temporal comparison grids
        for i, img in enumerate(self.val_grid_images):
            self.logger.experiment.add_image( # type: ignore
                f"Temporal_Comparison/batch_{i}",
                img,
                global_step=step
            ) 

        # Log grid images + scalars as a combined image panel
        grid_imgs = [row[1] for row in self.table_result_data]  # tensors (3,H,W)
        dice_vals = [row[2] for row in self.table_result_data]
        jac_vals = [row[3] for row in self.table_result_data]

        if grid_imgs:
            grid_panel = make_grid(torch.stack(grid_imgs), nrow=len(grid_imgs), padding=2, pad_value=1.0)
            self.logger.experiment.add_image("Grid/all", grid_panel, global_step=step) # type: ignore

        # Log per-sample scalars
        for row in self.table_result_data:
            sample_id, _, dice, nb_jac_neg = row
            self.logger.experiment.add_scalar(f"Dice/{sample_id}", dice, global_step=step) # type: ignore
            self.logger.experiment.add_scalar(f"JacNeg/{sample_id}", nb_jac_neg, global_step=step) # type: ignore

        mean_dice = float(np.mean(dice_vals))
        # Log mean dice and jac
        self.log("Val/mean_dice", mean_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Val/mean_jac_neg", float(np.mean(jac_vals)), on_step=False, on_epoch=True, prog_bar=True)

        self.logger.experiment.add_scalar("Val/mean_dice", mean_dice, global_step=step) # type: ignore
        self.logger.experiment.add_scalar("Val/mean_jac_neg", float(np.mean(jac_vals)), global_step=step) # type: ignore
 
        # Reset
        self.table_result_data = []
        self.val_grid_images = []

        if self.max_dice_score < mean_dice:
            self.max_dice_score = mean_dice
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_registration.pt"))

        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────────
    #  Test
    # ──────────────────────────────────────────────────────────────────────────

    def on_test_start(self) -> None:
        """Create output directories for images, parcellations, and flow fields."""
        # Create mri, seg and flows directories
        os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "parcellations"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "flows"), exist_ok=True)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """Register, warp, and save NIfTI outputs for every time-point of a test subject."""
        images, segs, ages = batch
        shape = images[0].shape[2:]
        scale_factor = torch.tensor(shape).to(self.device).view(1, 3, 1, 1, 1) * 1.
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)
        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        shape = images.shape[2:]
        initial_img = images[0:1].float()
        target_img = images[-1:].float()
        dices_subjects = []
        with torch.no_grad():
            all_phi, _ = self(initial_img, target_img, ages, ages[-1], grid)
        all_phi = all_phi.detach()
        grid_voxel = (grid + 1.) / 2. * scale_factor
        subject = self.trainer.test_dataloaders.dataset.get_subject(batch_idx) # type: ignore
        affine = subject.image.affine
        reverse_transform = tio.transforms.CropOrPad(subject.image.shape[1:])
        initial_seg = F.one_hot(segs[:, 0].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3)
        for idx in range(0, images.shape[0]):
            phi = all_phi[idx]
            df = phi - grid_voxel
            warped = registration.warp(images[0:1].float(), df)
            warped_seg = registration.warp(initial_seg.to(self.device).float(), df)
            warped_seg = torch.argmax(warped_seg, dim=1).detach()
            image = reverse_transform(tio.ScalarImage(tensor=warped.cpu().squeeze(0).float()))
            image.affine = affine
            image.save(os.path.join(self.save_dir, "images", f"subject_{batch_idx}_time_{idx:03d}.nii.gz"))

            parcellation = reverse_transform(tio.LabelMap(tensor=warped_seg.cpu().squeeze(0).float()))
            parcellation.affine = affine
            parcellation.save(os.path.join(self.save_dir, "parcellations", f"subject_{batch_idx}_time_{idx:03d}_seg.nii.gz"))

            df_image = reverse_transform(tio.ScalarImage(tensor=df.cpu().squeeze(0).float()))
            df_image.affine = affine
            df_image.save(os.path.join(self.save_dir, "flows", f"subject_{batch_idx}_time_{idx:03d}_flow.nii.gz"))

            if idx != 0:
                pred_label = F.one_hot(warped_seg.cpu().long(), num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3)
                dices_subjects.append(np.mean(self.seg_metrics(pred_label, F.one_hot(segs[:, idx].squeeze(0).cpu().long(),
                                                                             num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3).cpu())))
            print(f"Subject {batch_idx} : mean dice {np.mean(dices_subjects)}")
            del warped, warped_seg, phi, pred_label
            torch.cuda.empty_cache()

        del all_phi, df
        torch.cuda.empty_cache()

    def on_test_epoch_end(self) -> None:
        """Print final evaluation metrics and save the last model checkpoint."""
        print("Test epoch ended. Computing evaluation metrics...")
        print("Average Dice scores:", self.seg_metrics.compute())
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "saved_model.pt"))
        torch.cuda.empty_cache()
