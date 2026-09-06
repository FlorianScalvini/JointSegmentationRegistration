import json
from copy import deepcopy
from matplotlib.pyplot import grid
from pytest import param
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import monai
from model.unet import Unet
from model.neural_ode import LongitudinalODERegistration
import torchio as tio   
import utils.losses as losses
import utils.utils as utils
import utils.registration as registration
import os
import utils.utils as utils
import utils.losses as losses
import utils.visualize as visualize
import utils.registration as registration
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np
import nibabel as nib


def model_labels_to_raw(labels):
    """Restore raw label IDs: 0 stays 0 and model labels 1..3 become 2..4."""
    return torch.where(labels == 0, labels, labels + 1)


class PLJointRegistrationSegmentation(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-4, save_dir="", lambda_seg=1, lambda_reg=0.001, lambda_sim=0.0, lambda_jac:float = 0.000001, lambda_anchor:float = 1, ema_decay:float = 0.99, pretrain_registration_epochs:int = 0, segmentation_method:str = "temporal_neighbors", t0: float = 0.0, tn: float = 1.0, shape=[192, 224, 192], step_time=0.1, weight=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.automatic_optimization = False

        # Initialize the registration and segmentation networks
        self.registration = LongitudinalODERegistration(shape=shape, step_time=step_time)
        self.segmentation = Unet(in_channels=1, channels=[16, 32, 64, 128, 256], out_channels=num_classes, final_activation=None)
        self.segmentation_ema = deepcopy(self.segmentation)
        self.segmentation_ema.requires_grad_(False)
        self.segmentation_ema.eval()
        # Hyperparameters 
        self.lambda_reg = lambda_reg
        self.lambda_sim = lambda_sim
        self.lambda_seg = lambda_seg
        self.lambda_jac = lambda_jac
        self.lambda_anchor = lambda_anchor
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must satisfy 0 <= ema_decay < 1")
        self.ema_decay = ema_decay
        if segmentation_method not in {"temporal_neighbors", "t0_anchor"}:
            raise ValueError(
                "segmentation_method must be 'temporal_neighbors' or 't0_anchor'"
            )
        self.segmentation_method = segmentation_method
        self.num_classes = num_classes
        self.t0 = t0
        self.tn = tn
        # Loss functions and metrics
        self.loss_sim = monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=21)
        self.loss_reg = losses.Grad3d('l2')
        self.loss_seg = nn.MSELoss()
        self.loss_segmentation = monai.losses.DiceCELoss(
            to_onehot_y=False, softmax=True, include_background=True
        )
        self.loss_temporal = monai.losses.DiceLoss(
            to_onehot_y=False, softmax=False, include_background=True
        )
        self.seg_metrics_seg = monai.metrics.DiceMetric()
        self.seg_metrics_reg = monai.metrics.DiceMetric()
        if pretrain_registration_epochs < 0:
            raise ValueError("pretrain_registration_epochs must be non-negative")
        self.pretrain_registration_epochs = pretrain_registration_epochs
        # Logging and tracking best performance
        self.save_dir = save_dir
        self.max_dice_score = float('-inf')
        self.max_registration_dice_score = float('-inf')
        self.val_grid_images = []
        self.table_result_data = []
        self.registration_weight_loss = weight
        os.makedirs(os.path.join(self.save_dir, "segmentations"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "segmentations_errormaps"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "registration_parcellations"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "registration_images"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "registration_flows"), exist_ok=True)


    def configure_optimizers(self):
        opt_registration = torch.optim.Adam(self.registration.parameters(), lr=self.learning_rate)
        opt_segmentation = torch.optim.Adam(self.segmentation.parameters(), lr=self.learning_rate)
        return [opt_registration, opt_segmentation]

    @torch.no_grad()
    def _update_segmentation_ema(self) -> None:
        """Update the frozen pseudo-label teacher from the student weights."""
        for ema_parameter, parameter in zip(
            self.segmentation_ema.parameters(), self.segmentation.parameters()
        ):
            ema_parameter.lerp_(parameter, 1.0 - self.ema_decay)

        # Keep non-trainable buffers (if any) synchronized as well.
        for ema_buffer, buffer in zip(
            self.segmentation_ema.buffers(), self.segmentation.buffers()
        ):
            ema_buffer.copy_(buffer)

    @torch.no_grad()
    def sync_segmentation_ema(self) -> None:
        """Reset the teacher to the current student, e.g. after preloading."""
        self.segmentation_ema.load_state_dict(self.segmentation.state_dict())
        self.segmentation_ema.requires_grad_(False)
        self.segmentation_ema.eval()

    def on_load_checkpoint(self, checkpoint) -> None:
        """Allow older checkpoints without EMA weights to remain loadable."""
        self.max_dice_score = checkpoint.get("best_segmentation_dice", float('-inf'))
        self.max_registration_dice_score = checkpoint.get("best_registration_dice", float('-inf'))
        state_dict = checkpoint.get("state_dict", {})
        ema_prefix = "segmentation_ema."
        if not any(key.startswith(ema_prefix) for key in state_dict):
            student_prefix = "segmentation."
            for key, value in list(state_dict.items()):
                if key.startswith(student_prefix):
                    state_dict[ema_prefix + key[len(student_prefix):]] = value.clone()


    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["best_segmentation_dice"] = self.max_dice_score
        checkpoint["best_registration_dice"] = self.max_registration_dice_score

    def forward(self, x):
        raise NotImplementedError("Forward pass is integrated into the training step for joint optimization.")

    
    def forward_registration(self, initial_img, target_img, target_age, ages, grid):
        shape = initial_img.shape[2:]
        # align_corners=True maps [-1, 1] exactly onto voxel indices [0, N-1].
        scale_factor = (torch.tensor(shape, device=self.device, dtype=grid.dtype) - 1).view(1, 3, 1, 1, 1)
        all_phi, loss_reg, loss_jac = self.registration(
            initial_img, target_img, ages, target_age, grid
        )
        all_phi = (all_phi + 1.) / 2. * scale_factor
        return all_phi, loss_reg, loss_jac



    def _segmentation_step_temporal_neighbors(self, batch, batch_idx):
        _, opt_seg = self.optimizers()  # type: ignore
        images, _, ages, pretrain_mri, pretrain_seg, *_ = batch
        shape = images[0].shape[2:]
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)
        voxel_scale = (torch.tensor(shape, device=self.device, dtype=grid.dtype) - 1).view(1, 3, 1, 1, 1)
        grid_voxel = (grid + 1.) / 2. * voxel_scale

        images = images.squeeze(0)
        ages   = ages.squeeze(0).to(self.device)

        temporal_loss_sum = torch.tensor(0.0, device=self.device)
        supervised_loss_sum = torch.tensor(0.0, device=self.device)
        total_loss_sum = torch.tensor(0.0, device=self.device)
        for i in range(images.shape[0]):
            sequence_logits = self.segmentation(images[i:i+1].float())
            sequence_probability = torch.softmax(sequence_logits, dim=1)
            loss_temporal = sequence_logits.sum() * 0.0
            num_temporal_neighbors = 0

            window_start = max(0, i - 2)
            window_end = min(images.shape[0], i + 3)
            for k in range(window_start, window_end):
                if k == i:
                    continue
                if k < i:
                    trajectory_images = images[k:i + 1]
                    trajectory_ages = ages[k:i + 1]
                else:
                    # Reverse the slice so that image k is the source and
                    # image i is always the target of the registration.
                    trajectory_images = torch.flip(
                        images[i:k + 1], dims=[0]
                    )
                    trajectory_ages = torch.flip(
                        ages[i:k + 1], dims=[0]
                    )

                initial_img = trajectory_images[0:1].float()
                target_img = trajectory_images[-1:].float()
                with torch.no_grad():
                    all_phi, _, _ = self.forward_registration(
                        initial_img,
                        target_img,
                        trajectory_ages[-1],
                        trajectory_ages,
                        grid,
                    )
                    self.segmentation_ema.eval()
                    neighbour_probability = torch.softmax(
                        self.segmentation_ema(initial_img), dim=1
                    )
                    seg_warped = registration.warp(
                        neighbour_probability,
                        all_phi[-1] - grid_voxel,
                    )
                    seg_warped = seg_warped / seg_warped.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1e-8)
                loss_temporal = loss_temporal + self.loss_temporal(
                    sequence_probability, seg_warped
                )
                num_temporal_neighbors += 1

            if num_temporal_neighbors > 0:
                loss_temporal = loss_temporal / num_temporal_neighbors

            # The labeled sample returned alongside this sequence anchors the
            # class semantics and prevents constant-mask collapse.
            pretrain_logits = self.segmentation(
                pretrain_mri.float().to(self.device)
            )
            pretrain_target = F.one_hot(
                pretrain_seg.squeeze(0).long(), num_classes=self.num_classes
            ).permute(0, 4, 1, 2, 3).float().to(self.device)
            loss_supervised = self.loss_segmentation(pretrain_logits, pretrain_target)

            loss = loss_temporal + self.lambda_anchor * loss_supervised
            opt_seg.zero_grad()  # type: ignore
            self.manual_backward(loss)
            self.clip_gradients(
                opt_seg, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )
            opt_seg.step()  # type: ignore
            self._update_segmentation_ema()

            temporal_loss_sum += loss_temporal.detach()
            supervised_loss_sum += (self.lambda_anchor * loss_supervised).detach()
            total_loss_sum += loss.detach()
            torch.cuda.empty_cache()

        num_timepoints = images.shape[0]
        self.log(
            "train/segmentation/loss", total_loss_sum / num_timepoints,
            on_step=False, on_epoch=True, prog_bar=True,
        )
        self.log_dict(
            {
                "train/segmentation/temporal_loss": temporal_loss_sum / num_timepoints,
                "train/segmentation/supervised_loss": supervised_loss_sum / num_timepoints,
            },
            on_step=False, on_epoch=True, prog_bar=False,
        )

    def _segmentation_step_t0_anchor(self, batch, batch_idx):
        """Train segmentation from the GT at t0 transported along the ODE."""
        _, opt_seg = self.optimizers()  # type: ignore
        images, segmentations, ages, pretrain_mri, pretrain_seg, *_ = batch
        images = images.squeeze(0)
        segmentations = segmentations.squeeze(0).to(self.device)
        ages = ages.squeeze(0).to(self.device)

        shape = images.shape[2:]
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)
        voxel_scale = (
            torch.tensor(shape, device=self.device, dtype=grid.dtype) - 1
        ).view(1, 3, 1, 1, 1)
        grid_voxel = (grid + 1.0) / 2.0 * voxel_scale

        t0_segmentation = F.one_hot(
            segmentations[0].long(), num_classes=self.num_classes
        ).permute(0, 4, 1, 2, 3).float()

        with torch.no_grad():
            all_phi, _, _ = self.forward_registration(
                images[0:1].float(),
                images[-1:].float(),
                ages[-1],
                ages,
                grid,
            )

        transported_loss_sum = torch.tensor(0.0, device=self.device)
        supervised_loss_sum = torch.tensor(0.0, device=self.device)
        total_loss_sum = torch.tensor(0.0, device=self.device)

        for index in range(images.shape[0]):
            student_logits = self.segmentation(images[index:index + 1].float())
            displacement = all_phi[index] - grid_voxel
            with torch.no_grad():
                transported = registration.warp(t0_segmentation, displacement)
                transported = transported.argmax(dim=1)
                transported = F.one_hot(
                    transported.long(), num_classes=self.num_classes
                ).permute(0, 4, 1, 2, 3).float()

            loss_transported = self.loss_segmentation(
                student_logits, transported
            )
            pretrain_logits = self.segmentation(
                pretrain_mri.float().to(self.device)
            )
            pretrain_target = F.one_hot(
                pretrain_seg.squeeze(0).long(), num_classes=self.num_classes
            ).permute(0, 4, 1, 2, 3).float().to(self.device)
            loss_supervised = self.loss_segmentation(
                pretrain_logits, pretrain_target
            )
            loss = loss_transported + self.lambda_anchor * loss_supervised

            opt_seg.zero_grad()  # type: ignore
            self.manual_backward(loss)
            self.clip_gradients(
                opt_seg, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )
            opt_seg.step()  # type: ignore
            self._update_segmentation_ema()

            transported_loss_sum += loss_transported.detach()
            supervised_loss_sum += (
                self.lambda_anchor * loss_supervised
            ).detach()
            total_loss_sum += loss.detach()

        number_of_timepoints = images.shape[0]
        self.log(
            "train/segmentation/loss",
            total_loss_sum / number_of_timepoints,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(
            {
                "train/segmentation/t0_transport_loss": (
                    transported_loss_sum / number_of_timepoints
                ),
                "train/segmentation/supervised_loss": (
                    supervised_loss_sum / number_of_timepoints
                ),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def _segmentation_step(self, batch, batch_idx):
        if self.segmentation_method == "t0_anchor":
            self._segmentation_step_t0_anchor(batch, batch_idx)
        else:
            self._segmentation_step_temporal_neighbors(batch, batch_idx)

    def _registration_loss(
        self, images, ages, segs, target_index, grid, scale_factor
    ):
        """Compute similarity, segmentation, gradient and Jacobian losses."""
        loss_sim = torch.tensor(0.0, device=self.device)
        loss_seg = torch.tensor(0.0, device=self.device)
        initial_img = images[0:1].float()
        target_img = images[target_index:target_index + 1].float()
        initial_seg = segs[0].float().to(self.device)
        all_phi, loss_reg, loss_jac = self.forward_registration(
            initial_img, target_img, ages[target_index], ages, grid
        )
        
        grid_voxel = (grid + 1.) / 2. * scale_factor

        for idx in range(1, all_phi.shape[0]):
            phi = all_phi[idx]
            df = phi - grid_voxel
            if self.lambda_sim > 0:
                warped = registration.warp(initial_img, df)
                loss_sim += self.loss_sim(warped, images[idx:idx + 1].float())
                del warped
            if self.lambda_seg > 0:
                warped_seg = registration.warp(initial_seg, df)
                loss_seg += self.loss_seg(
                    warped_seg, segs[idx].to(self.device).float()
                )
            else:
                warped_seg = None
            if warped_seg is not None:
                del warped_seg
            del phi, df

        num_steps = all_phi.shape[0] - 1
        loss_seg = loss_seg / num_steps
        loss_sim = loss_sim / num_steps
        trajectory_duration = torch.abs(ages[-1] - ages[0]).clamp_min(1e-8)
        loss_reg = loss_reg / trajectory_duration
        loss_jac = loss_jac / trajectory_duration
        loss = (
            self.lambda_sim * loss_sim
            + self.lambda_seg * loss_seg
            + self.lambda_reg * loss_reg
            + self.lambda_jac * loss_jac
        )
        components = {
            'loss_sim': (self.lambda_sim * loss_sim).detach(),
            'loss_seg': (self.lambda_seg * loss_seg).detach(),
            'loss_grad': (self.lambda_reg * loss_reg).detach(),
            'loss_jac': (self.lambda_jac * loss_jac).detach(),
        }
        return loss, components

    def self_training_registration_step(self, batch, batch_idx):
        """Train trajectories towards the fixed endpoint in each direction."""
        images, _, ages, _, _ = batch
        shape = images[0].shape[2:]
        # Must match the scale used in forward_registration (shape - 1, align_corners=True)
        scale_factor = (
            torch.tensor(shape, device=self.device, dtype=torch.float32) - 1
        ).view(1, 3, 1, 1, 1)
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)

        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        with torch.no_grad():
            self.segmentation_ema.eval()
            segs = torch.stack([
                F.one_hot(
                    self.segmentation_ema(images[i:i + 1].float()).argmax(dim=1),
                    num_classes=self.num_classes,
                ).permute(0, 4, 1, 2, 3).float()
                for i in range(images.shape[0])
            ], dim=0).to(self.device)

        optimizer, _ = self.optimizers()
        optimizer.zero_grad()  # type: ignore

        num_timepoints = images.shape[0]
        if num_timepoints < 2:
            raise ValueError("registration training requires at least two images")

        component_sums = {
            "loss_sim": torch.tensor(0.0, device=self.device),
            "loss_seg": torch.tensor(0.0, device=self.device),
            "loss_grad": torch.tensor(0.0, device=self.device),
            "loss_jac": torch.tensor(0.0, device=self.device),
        }
        loss_sum = torch.tensor(0.0, device=self.device)
        num_trajectories = 2 * (num_timepoints - 1)

        for i in range(num_timepoints):
            for direction in ("forward", "reverse"):
                if direction == "forward":
                    if i == num_timepoints - 1:
                        continue
                    trajectory_images = images[i:]
                    trajectory_ages = ages[i:]
                    trajectory_segs = segs[i:]
                else:
                    if i == 0:
                        continue
                    trajectory_images = torch.flip(images[:i + 1], dims=[0])
                    trajectory_ages = torch.flip(ages[:i + 1], dims=[0])
                    trajectory_segs = torch.flip(segs[:i + 1], dims=[0])

                # The conditioning target is always the final image of the
                # directed trajectory. All other images are intermediate.


                trajectory_loss, trajectory_components = self._registration_loss(
                    trajectory_images,
                    trajectory_ages,
                    trajectory_segs,
                    trajectory_images.shape[0] - 1,
                    grid,
                    scale_factor,
                )
                if not torch.isfinite(trajectory_loss):
                    raise FloatingPointError(f"Non-finite registration loss at batch {batch_idx}, trajectory {i} {direction}")
                self.manual_backward(trajectory_loss / num_trajectories)
                loss_sum += trajectory_loss.detach()
                for name, value in trajectory_components.items():
                    component_sums[name] += value

        loss = loss_sum / num_trajectories
        components = {
            name: value / num_trajectories
            for name, value in component_sums.items()
        }

        # Clip the complete average gradient, after Lightning unscales it.
        self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in self.registration.parameters()):
            raise FloatingPointError(f"Non-finite registration gradient at batch {batch_idx}")
        optimizer.step()  # type: ignore

        self.log(
            "train/registration/loss", loss.detach(),
            on_step=False, on_epoch=True, prog_bar=True,
        )
        self.log_dict(
            {
                "train/registration/loss_similarity": components["loss_sim"],
                "train/registration/loss_segmentation": components["loss_seg"],
                "train/registration/loss_smoothness": components["loss_grad"],
                "train/registration/loss_jacobian": components["loss_jac"],
            },
            on_step=False, on_epoch=True, prog_bar=False,
        )
        torch.cuda.empty_cache()
    


    def training_step(self, batch, batch_idx):
        print(batch_idx)
        cotraining_epoch = self.current_epoch - self.pretrain_registration_epochs
        train_segmentation = (
            cotraining_epoch >= 0 and cotraining_epoch % 2 == 0
        )
        if train_segmentation:
            self.segmentation.train()
            self.registration.eval()
            for param in self.segmentation.parameters():
                param.requires_grad = True
            for param in self.registration.parameters():
                param.requires_grad = False
            self._segmentation_step(batch, batch_idx)
        else:
            self.segmentation.eval()
            self.registration.train()
            for param in self.segmentation.parameters():
                param.requires_grad = False
            for param in self.registration.parameters():
                param.requires_grad = True
            self.self_training_registration_step(batch, batch_idx)

        
    def on_train_epoch_end(self) -> None:
        torch.cuda.empty_cache()  # ← add this
        torch.save(self.registration.state_dict(), os.path.join(self.save_dir, "last_registration.pt"))
        torch.save(self.segmentation.state_dict(), os.path.join(self.save_dir, "last_segmentation.pt"))

    def on_validation_epoch_start(self) -> None:
        self.seg_metrics_seg.reset()
        self.scores = {}

    @staticmethod
    def _save_displacement_nifti(flow_ras_mm, affine, output_path):
        """Save a Slicer-compatible NIfTI displacement vector field."""
        if flow_ras_mm.ndim != 4 or flow_ras_mm.shape[0] != 3:
            raise ValueError(f"flow must have shape (3,I,J,K), got {tuple(flow_ras_mm.shape)}")
        data = flow_ras_mm.detach().cpu().permute(1, 2, 3, 0).unsqueeze(3).numpy()
        image = nib.Nifti1Image(data.astype(np.float32, copy=False), affine)
        image.header.set_intent(1006, name="displacement")
        nib.save(image, output_path)


    def validation_step(self, batch, batch_idx):
        images, segs, ages = batch
        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        segs = segs.squeeze(0)
        subject_scores = []

        with torch.no_grad():
            for idx in range(images.shape[0]):
                affine = self.trainer.val_dataloaders.dataset.get_subject(batch_idx, idx).image.affine
                preds_seg = self.segmentation(images[idx:idx + 1].float())
                preds_seg = torch.argmax(preds_seg, dim=1)
                seg_i = F.one_hot(segs[idx].long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
                tio.LabelMap(tensor=model_labels_to_raw(preds_seg).cpu(), affine=affine).save(os.path.join(self.save_dir, "segmentations", f"pred_seg_sample{batch_idx}_time{idx}.nii.gz"))
                tio.LabelMap(tensor=((segs[idx] != preds_seg)*1.0).cpu(), affine=affine).save(os.path.join(self.save_dir, "segmentations_errormaps", f"segmentation_sample{batch_idx}_time{idx}.nii.gz"))
                preds_seg = F.one_hot(preds_seg, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
                self.seg_metrics_seg(preds_seg.cpu(), seg_i.cpu())
                subject_scores.append(self.seg_metrics_seg.get_buffer()[-1].numpy().tolist())
        self.scores[batch_idx] = subject_scores
        shape = images.shape[2:]
        scale_factor = (torch.tensor(shape, device=self.device, dtype=torch.float32) - 1).view(1, 3, 1, 1, 1)
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)
    
        grid_voxel = (grid + 1.) / 2. * scale_factor

        all_registered = []
        all_targets = []
        all_segs = []
        initial_img = images[0:1].float()
        target_img = images[-1:].float()
        initial_seg = F.one_hot(segs[0:1].squeeze(0).cpu().long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
        with torch.no_grad():
            all_phi, _, _ = self.forward_registration(
                initial_img, target_img, ages[-1], ages, grid
            )
        all_phi = all_phi.detach()
        for idx in range(0, images.shape[0]):
            original_session = self.trainer.val_dataloaders.dataset.get_subject(batch_idx, idx)

            model_affine = original_session.image.affine
            phi = all_phi[idx]
            df = phi - grid_voxel
            warped = registration.warp(images[0:1].float(), df)
            warped_seg = registration.warp(initial_seg.to(self.device).float(), df)
            warped_seg = torch.argmax(warped_seg, dim=1).detach()
            tio.LabelMap(tensor=model_labels_to_raw(warped_seg).cpu(), affine=model_affine).save(os.path.join(self.save_dir, "registration_parcellations", f"segmentation_sample{batch_idx}_time{idx}.nii.gz"))
            tio.ScalarImage(tensor=warped.squeeze(0).cpu(), affine=model_affine).save(os.path.join(self.save_dir, "registration_images", f"image_sample{batch_idx}_time{idx}.nii.gz"))
            # The network predicts voxel-axis increments. Convert them to RAS-mm
            # vectors and preserve the model grid affine; do not treat a vector
            # field as three scalar channels during the reverse transform.
            flow_ijk = df.squeeze(0).cpu()
            model_linear = torch.as_tensor(model_affine[:3, :3], dtype=flow_ijk.dtype)
            flow_ras_mm = torch.einsum("rc,cijk->rijk", model_linear, flow_ijk)
            self._save_displacement_nifti(
                flow_ras_mm, model_affine,
                os.path.join(self.save_dir, "registration_flows", f"df_sample{batch_idx}_time{idx}.nii.gz"),
            )
            pred_label = F.one_hot(warped_seg.cpu().long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3)
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
                self.seg_metrics_reg(pred_label, F.one_hot(segs[idx].cpu().long(),num_classes=self.num_classes).permute(0, 4, 1, 2, 3).cpu())
                det_jac = utils.compute_jacobian_determinant_3d(df.cpu()).numpy()
                nb_jac_neg = int(np.sum(det_jac < 0))
                buffer = self.seg_metrics_reg.get_buffer()
                dice = float(buffer[-1].nanmean().item())
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
        # Compute and log mean Dice score for segmentation
        mean_dice_seg = self.seg_metrics_seg.aggregate().item()
        mean_dice = self.seg_metrics_reg.aggregate().item()
        self.seg_metrics_seg.reset()
        self.seg_metrics_reg.reset()
        self.log(
            "validation/segmentation/dice",
            mean_dice_seg,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        torch.cuda.empty_cache()
        if not self.trainer.sanity_checking and np.isfinite(mean_dice_seg) and self.max_dice_score < mean_dice_seg:
            self.max_dice_score = mean_dice_seg
            torch.save(self.segmentation.state_dict(), os.path.join(self.save_dir, "best_segmentation.pt"))
        if not self.trainer.sanity_checking and np.isfinite(mean_dice) and self.max_registration_dice_score < mean_dice:
            self.max_registration_dice_score = mean_dice
            torch.save(self.registration.state_dict(), os.path.join(self.save_dir, "best_registration.pt"))
        json.dump(self.scores, open(os.path.join(self.save_dir, "dice_scores.json"), "w"))
        if self.current_epoch == 0:
            json.dump(self.scores, open(os.path.join(self.save_dir, "pretrain_results.json"), "w"))
        
        # TensorBoard event files are append-only: reusing global_step=0 does
        # not replace an image and makes the log grow at every epoch.  Keep
        # only the latest visualizations as regular PNG files instead.
        for i, img in enumerate(self.val_grid_images):
            save_image(
                img,
                os.path.join(self.save_dir, f"latest_temporal_batch_{i}.png"),
            ) 

        # Log grid images + scalars as a combined image panel
        grid_imgs = [row[1] for row in self.table_result_data]  # tensors (3,H,W)
        jac_vals = [row[3] for row in self.table_result_data]

        if grid_imgs:
            grid_panel = make_grid(torch.stack(grid_imgs), nrow=len(grid_imgs), padding=2, pad_value=1.0)
            save_image(
                grid_panel,
                os.path.join(self.save_dir, "latest_deformation_grids.png"),
            )

        mean_jac_neg = float(np.mean(jac_vals)) if jac_vals else 0.0
        self.log(
            "validation/registration/dice",
            mean_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "validation/registration/negative_jacobians",
            mean_jac_neg,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
 
        # Reset
        self.table_result_data = []
        self.val_grid_images = []

        torch.cuda.empty_cache()
