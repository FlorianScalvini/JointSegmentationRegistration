import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import monai
from model.unet import Unet
from model.neural_ode import LongitudinalODERegistration
import utils.losses as losses
import utils.utils as utils
import utils.registration as registration
import random 
import os
import torchio as tio

class PLSegmentation(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.01, save_dir="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.automatic_optimization = False
        self.model = Unet(in_channels=1, channels=[16, 32, 64, 128, 256], out_channels=num_classes, final_activation=None)
        # Loss functions and metrics
        self.loss_seg = monai.losses.DiceCELoss(softmax=True)
        self.seg_metrics_train = monai.metrics.DiceMetric()
        self.seg_metrics_val = monai.metrics.DiceMetric()
        # Logging and tracking best performance
        self.save_dir = save_dir
        self.max_dice_score = 0


        
    def configure_optimizers(self):
        opt_segmentation = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler_segmentation = torch.optim.lr_scheduler.ExponentialLR(opt_segmentation, gamma=0.999)
        return [opt_segmentation], [lr_scheduler_segmentation]


    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        self.seg_metrics_train.reset()

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        images, segs, _ = batch
        images = images.squeeze(0)  
        segs = segs.squeeze(0)
        global_loss = torch.tensor(0.0, device=self.device)
        for i in range(images.shape[0]):
            seg_i_pred = self.model(images[i:i + 1].float())
            seg_gt = F.one_hot(segs[i].long(), num_classes=self.hparams.num_classes).permute(0, 4, 1, 2, 3).float()
            loss = self.loss_seg(seg_i_pred, seg_gt)
            global_loss += loss.item()
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            seg_i_hard = F.one_hot(torch.argmax(seg_i_pred, dim=1), num_classes=self.hparams.num_classes).permute(0, 4, 1, 2, 3).float()
            self.seg_metrics_train(seg_i_hard.cpu(), seg_gt.cpu())
        self.log('Train/Loss', global_loss / images.shape[0], on_step=False, on_epoch=True, prog_bar=True)
        # Segmentation steps, then registration steps


    def on_train_epoch_end(self) -> None:
        torch.cuda.empty_cache()  # ← add this
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "last_segmentation.pt"))
        self.log('Train/Dice', self.seg_metrics_train.aggregate().item(), on_step=False, on_epoch=True, prog_bar=True)
        self.seg_metrics_train.reset()

    def on_validation_epoch_start(self) -> None:
        self.seg_metrics_val.reset()

    def validation_step(self, batch, batch_idx):
        images, segs, _ = batch
        images = images.squeeze(0) 
        segs = segs.squeeze(0)
        for i in range(images.shape[0]):
            preds_seg = self.model(images[i:i + 1].float())
            preds_seg = torch.argmax(preds_seg, dim=1)
            tio.LabelMap(tensor=preds_seg.cpu()).save(os.path.join(self.save_dir, f"pred_seg_{batch_idx}_{i}.nii.gz"))
            preds_seg = F.one_hot(preds_seg.long(), num_classes=self.hparams.num_classes).permute(0, 4, 1, 2, 3).float()
            seg_i = F.one_hot(segs[i].long(), num_classes=self.hparams.num_classes).permute(0, 4, 1, 2, 3).float()
            self.seg_metrics_val(preds_seg.cpu(), seg_i.cpu())
            


    def on_validation_epoch_end(self) -> None:
        if self.seg_metrics_val.get_buffer().shape[0] == 0:
            mean_dice_seg = 0.0
        else:
            # Compute and log mean Dice score for segmentation
            mean_dice_seg = self.seg_metrics_val.get_buffer().mean().item()
            self.seg_metrics_val.reset()
            self.log('mDice_seg', mean_dice_seg, prog_bar=True, on_epoch=True)
            self.logger.experiment.add_scalar("Val/mDice_seg", mean_dice_seg, global_step=self.current_epoch) # type: ignore
    
        torch.cuda.empty_cache()
        if self.max_dice_score < mean_dice_seg:
            self.max_dice_score = mean_dice_seg
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_segmentation.pt"))