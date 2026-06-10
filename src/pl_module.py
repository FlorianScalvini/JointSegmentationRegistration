from matplotlib.pyplot import grid
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
import random 
import os
import utils.utils as utils
import utils.losses as losses
import utils.visualize as visualize
import utils.registration as registration
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
import numpy as np

class PLJointRegistrationSegmentation(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.01, save_dir="", lambda_seg=1, lambda_reg=0.001, lambda_sdf=1, lambda_sim=0.0, lambda_jac:float = 0.000001, shape=[192, 224, 192], step_time=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.automatic_optimization = False

        # Initialize the registration and segmentation networks
        self.registration = LongitudinalODERegistration(shape=shape, step_time=step_time)
        self.segmentation = Unet(in_channels=1, channels=[16, 32, 64, 128, 256], out_channels=num_classes, final_activation=None)

        # Hyperparameters 
        self.lambda_sdf = lambda_sdf
        self.lambda_reg = lambda_reg
        self.lambda_sim = lambda_sim
        self.lambda_seg = lambda_seg
        self.lambda_jac = lambda_jac

        # Loss functions and metrics
        self.loss_sim = monai.losses.LocalNormalizedCrossCorrelationLoss(kernel_size=21)
        self.loss_reg = losses.Grad3d('l2')
        self.loss_sdf = nn.L1Loss()
        self.loss_seg = nn.MSELoss()
        self.loss_jac = losses.NonDetJacobianPenalty()

        self.seg_metrics_seg = monai.metrics.DiceMetric()
        self.seg_metrics_reg = monai.metrics.DiceMetric()

        # Logging and tracking best performance
        self.save_dir = save_dir
        self.max_dice_score = 0
        self.val_grid_images = []
        self.table_result_data = []

        
    def configure_optimizers(self):
        opt_registration = torch.optim.Adam(self.registration.parameters(), lr=self.learning_rate)
        opt_segmentation = torch.optim.Adam(self.segmentation.parameters(), lr=self.learning_rate)

        lr_scheduler_registration = torch.optim.lr_scheduler.ExponentialLR(opt_registration, gamma=0.999)
        lr_scheduler_segmentation = torch.optim.lr_scheduler.ExponentialLR(opt_segmentation, gamma=0.999)
        return [opt_registration, opt_segmentation], [lr_scheduler_registration, lr_scheduler_segmentation]


    def forward(self, x):
        return NotImplementedError("Forward pass is integrated into the training step for joint optimization.") 

    
    def forward_registration(self, initial_img, target_img, target_age, ages, grid):
        all_phi, loss_reg = self.registration(initial_img, target_img, ages, target_age, grid)
        return all_phi, loss_reg
    
    def _training_segmentation_step(self, batch, batch_idx):
        _, optimizer = self.optimizers() # type: ignore
        images, _, ages = batch
        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        loss_total_seq = torch.tensor(0.0, device=self.device)
        shape = images.shape[2:]
        scale_factor = torch.tensor(shape).to(self.device).view(1, 3, 1, 1, 1) * 1.
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)

        for i in range(images.shape[0]):
            # Compute segmentation loss for image I against all other segmentations in the sequence registered to I
            loss = torch.tensor(0.0, device=self.device)
            seg_i_pred = self.segmentation(images[i:i + 1].float())
            for k in range(images.shape[0]):
                if i == k:
                    continue
                # Get deformation from k to i
                initial_img = images[i:i + 1].float()
                target_img = images[k:k + 1].float()
                agesIK = torch.Tensor([ages[i], ages[k]]).to(self.device)
                with torch.no_grad():
                    all_phi, _ = self.forward_registration(initial_img, target_img, ages[k], agesIK, grid)
                all_phi = all_phi.detach()
                grid_voxel = (grid + 1.) / 2. * scale_factor
                df = all_phi[-1] - grid_voxel
                warped_seg = registration.warp(seg_i_pred, df)
                # Get segmentation of image k and warp it to i
                seg_k_pred = self.segmentation(images[k:k + 1].float())
                loss += self.loss_seg(warped_seg, seg_k_pred)
            loss_total_seq += loss / (images.shape[0] - 1)  # Average over all pairs for image i
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()

        self.log_dict({
            'Segmentation/loss': loss_total_seq.item()
        }, on_step=False, on_epoch=True, prog_bar=True)
    

    def _training_registration_step(self, batch, batch_idx):
        images, _, ages = batch
        optimizer, _ = self.optimizers() # type: ignore
        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        shape = images.shape[2:]
        scale_factor = torch.tensor(shape).to(self.device).view(1, 3, 1, 1, 1) * 1.
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)
        # Initialise losses to zero
        segs = []
        with torch.no_grad():
            for i in range(images.shape[0]):
                seg_i_pred = self.segmentation(images[i:i + 1].float())
                segs.append(torch.argmax(seg_i_pred, dim=1).float().to(self.device))

        for i in range(images.shape[0]):
            loss_sim = torch.tensor(0.0, device=self.device)
            loss_seg = torch.tensor(0.0, device=self.device)
            loss_reg_tot = torch.tensor(0.0, device=self.device)
            loss_jac = torch.tensor(0.0, device=self.device)
            img_i = images[i:i + 1].float()
            seg_i_pred = F.one_hot(segs[i].long(), num_classes=-1).permute(0, 4, 1, 2, 3).float()
            grid_voxel = (grid + 1.) / 2. * scale_factor
            # Forward registration from i to all other time points k, compute losses for each pair (i,k) and accumulate
            if ages[i:].shape[0] > 1:
                subset_imgs = images[i:]
                subset_ages = ages[i:]
                subset_segs = segs[i:]
                all_phi, loss_reg = self.forward_registration(img_i, subset_imgs[-1], subset_ages[-1], subset_ages, grid)
                loss_reg_tot += loss_reg
                all_phi = (all_phi + 1.) / 2. * scale_factor
                for t in range(1, subset_imgs.shape[0]):
                    df = all_phi[t] - grid_voxel
                    if self.lambda_sim > 0:
                        warped = registration.warp(img_i, df)
                        loss_sim += self.loss_sim(warped, subset_imgs[t])
                        del warped

                    if self.lambda_seg > 0:
                        warped_seg = registration.warp(seg_i_pred.float().to(self.device), df)
                        seg_t_pred = F.one_hot(subset_segs[t].long(), num_classes=-1).permute(0, 4, 1, 2, 3).float().to(self.device)
                        loss_seg += self.loss_seg(warped_seg, seg_t_pred)
                        del warped_seg, seg_t_pred
                    if self.lambda_jac > 0:
                        loss_jac += self.loss_jac(df)
                    del df
            # Backward registration from i to all other time points k, compute losses for each pair (i,k) and accumulate
            if ages[:i + 1].shape[0] > 1:
                    subset_imgs = images[:i + 1]
                    subset_imgs = torch.flip(subset_imgs, dims=[0])
                    subset_ages = ages[:i + 1]
                    subset_ages = torch.flip(subset_ages, dims=[0])
                    subset_segs = segs[:i + 1]
                    subset_segs.reverse()
                    all_phi, loss_reg = self.forward_registration(img_i, subset_imgs[-1], subset_ages[-1], subset_ages, grid)
                    loss_reg_tot += loss_reg
                    all_phi = (all_phi + 1.) / 2. * scale_factor
                    for t in range(1, subset_imgs.shape[0]):
                        df = all_phi[t] - grid_voxel
                        if self.lambda_sim > 0:
                            warped = registration.warp(img_i, df)
                            loss_sim += self.loss_sim(warped, subset_imgs[t])
                            del warped

                        if self.lambda_seg > 0:
                            warped_seg = registration.warp(seg_i_pred.float().to(self.device), df)
                            seg_t_pred = F.one_hot(subset_segs[t].long(), num_classes=-1).permute(0, 4, 1, 2, 3).float().to(self.device)
                            loss_seg += self.loss_seg(warped_seg, seg_t_pred)
                            del warped_seg, seg_t_pred
                        if self.lambda_jac > 0:
                            loss_jac += self.loss_jac(df)
                        del df
                    del all_phi, grid_voxel
                    torch.cuda.empty_cache()
            nb_element = images.shape[0] - 1
            loss_seg = loss_seg / nb_element
            loss_sim = loss_sim / nb_element
            loss_jac = loss_jac / nb_element
            loss_reg = loss_reg / ((ages[-1] - ages[0]) / self.registration.step_time) # Normalize by number of integration steps, not number of images

            loss =  self.lambda_sim * loss_sim + self.lambda_seg * loss_seg  + self.lambda_reg * loss_reg + self.lambda_jac * loss_jac 
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            self.log_dict({
                'Registration/loss': loss.item(),
                'Registration/loss_sim': (self.lambda_sim * loss_sim).item(),
                'Registration/loss_seg': (self.lambda_seg * loss_seg).item(),
                'Registration/loss_jac': (self.lambda_jac * loss_jac).item()
            }, on_step=False, on_epoch=True, prog_bar=True)
            del loss, loss_sim, loss_reg
        # ── always flush at end of step ──
        torch.cuda.empty_cache()


    def training_step(self, batch, batch_idx):
        # Segmentation steps, then registration steps
        if self.current_epoch > 1000:
            self._training_segmentation_step(batch, batch_idx)
        self._training_registration_step(batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        torch.cuda.empty_cache()  # ← add this
        torch.save(self.registration.state_dict(), os.path.join(self.save_dir, "last_registration.pt"))
        torch.save(self.segmentation.state_dict(), os.path.join(self.save_dir, "last_segmentation.pt"))

    def on_validation_epoch_start(self) -> None:
        self.seg_metrics_seg.reset()

    def validation_step(self, batch, batch_idx):
        images, segs, ages = batch
        images = images.squeeze(0)
        ages = ages.squeeze(0).to(self.device)
        segs = segs.squeeze(0)
        for i in range(images.shape[0]):
            preds_seg = self.segmentation(images[i:i + 1].float())
            preds_seg = torch.argmax(preds_seg, dim=1)
            seg_i = F.one_hot(segs[i].long(), num_classes=-1).permute(0, 4, 1, 2, 3).float()
            pred_seg_i = F.one_hot(preds_seg, num_classes=-1).permute(0, 4, 1, 2, 3).float()
            tio.LabelMap(tensor=pred_seg_i.squeeze().cpu()).save(os.path.join(self.save_dir, f"pred_seg_sample{batch_idx}_time{i}.nii.gz"))
            self.seg_metrics_seg(pred_seg_i.cpu(), seg_i.cpu())


        shape = images.shape[2:]
        scale_factor = torch.tensor(shape).to(self.device).view(1, 3, 1, 1, 1) * 1.
        grid = registration.generate_grid3d_tensor(shape).unsqueeze(0).to(self.device)

        initial_img = images[0:1].float()
        target_img = images[-1:].float()
        with torch.no_grad():
            all_phi, _ = self.forward_registration(initial_img, target_img, ages[-1], ages, grid)
        all_phi = all_phi.detach()
        all_phi = (all_phi + 1.) / 2. * scale_factor
        grid_voxel = (grid + 1.) / 2. * scale_factor
        all_registered = []
        all_targets = []
        all_segs = []

        initial_seg = F.one_hot(segs[0:1].squeeze(0).cpu().long(), num_classes=-1).permute(0, 4, 1, 2, 3)
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
                self.seg_metrics_reg(pred_label, F.one_hot(segs[idx].cpu().long(),num_classes=initial_seg.shape[1]).permute(0, 4, 1, 2, 3).cpu())
                det_jac = utils.compute_jacobian_determinant_3d(df.cpu()).numpy()
                nb_jac_neg = int(np.sum(det_jac < 0))
                buffer = self.seg_metrics_reg.get_buffer()
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
        # Compute and log mean Dice score for segmentation
        mean_dice_seg = self.seg_metrics_seg.aggregate().item()
        self.seg_metrics_seg.reset()
        self.seg_metrics_reg.reset()
        self.log('Validation/mDice_seg', mean_dice_seg, prog_bar=True, on_epoch=True)
        self.logger.experiment.add_scalar("Val/mDice_seg", mean_dice_seg, global_step=self.current_epoch) # type: ignore
        torch.cuda.empty_cache()
        if self.max_dice_score < mean_dice_seg:
            self.max_dice_score = mean_dice_seg
            torch.save(self.registration.state_dict(), os.path.join(self.save_dir, "best_registration.pt"))
            torch.save(self.segmentation.state_dict(), os.path.join(self.save_dir, "best_segmentation.pt"))

        step = self.current_epoch

        # Log temporal comparison grids
        for i, img in enumerate(self.val_grid_images):
            self.logger.experiment.add_image( # type: ignore
                f"Temporal_Comparison/batch_{i}",
                img,
                global_step=0
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

        torch.cuda.empty_cache()