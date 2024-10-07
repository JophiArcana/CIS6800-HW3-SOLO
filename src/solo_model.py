from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from backbone import Resnet50Backbone
from solo_head import SOLOHead


class SOLO(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = Resnet50Backbone()
        self.head = SOLOHead(4)

    def forward(self, img: torch.tensor, evaluate: bool = False):
        fpn_feat_list = [v.detach() for v in self.backbone(img).values()]
        return self.head(fpn_feat_list, evaluate=evaluate)

    def training_step(self, batch, batch_idx: int):
        if self.trainer.current_epoch in (27, 33):
            self.lr_schedulers().step()

        img, label_list, mask_list, bbox_list = batch

        cate_pred_list, ins_pred_list = self.forward(img, evaluate=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.head.target(ins_pred_list, bbox_list, label_list, mask_list)

        loss = self.head.loss(cate_pred_list, ins_pred_list, ins_gts_list, ins_ind_gts_list, cate_gts_list)

        # self.train_loss.append(loss.item())
        self.log("training_loss", loss.item(), on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        img, label_list, mask_list, bbox_list = batch

        cate_pred_list, ins_pred_list = self.forward(img, evaluate=True)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = self.head.target(ins_pred_list, bbox_list, label_list,mask_list)

        loss = self.head.loss(cate_pred_list, ins_pred_list, ins_gts_list, ins_ind_gts_list, cate_gts_list)
        # self.val_loss.append(loss.item())
        self.log("validation_loss", loss.item(), on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.SGD(self.head.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}




