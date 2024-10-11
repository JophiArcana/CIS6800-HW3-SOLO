from copy import deepcopy
from functools import partial
from types import MappingProxyType

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import center_of_mass

from backbone import *
from dataset import *

from settings import DEVICE


def conv_gn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_groups=32, bias=False):
    """Helper function to create a Conv2d -> GroupNorm -> ReLU layer."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.GroupNorm(num_groups, out_channels),
        nn.ReLU(inplace=False)
    )


class SOLOHead(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels=256,
        seg_feat_channels=256,
        stacked_convs=7,
        strides=(8, 8, 16, 32, 32),
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        epsilon=0.2,
        num_grids=(40, 36, 24, 16, 12),
        cate_down_pos=0,
        with_deform=False,
        mask_loss_cfg=MappingProxyType({"weight": 3}),
        cate_loss_cfg=MappingProxyType({"gamma": 2, "alpha": 0.25, "weight": 1}),
        postprocess_cfg=MappingProxyType({
            "cate_thresh": 0.2,
            "ins_thresh": 0.5,
            "pre_NMS_num": 50,          # ??????
            "keep_instance": 2,
            "IoU_thresh": 0.5           # ??????
        })
    ):
        super(SOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.epsilon = epsilon
        self.cate_down_pos = cate_down_pos
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform

        self.mask_loss_cfg = mask_loss_cfg
        self.cate_loss_cfg = cate_loss_cfg
        self.postprocess_cfg = postprocess_cfg

        # Initialize layers and weights
        self._init_layers()
        self._init_weights()

        # Check consistency
        assert len(self.ins_head) == self.stacked_convs
        assert len(self.cate_head) == self.stacked_convs
        assert len(self.ins_out_list) == len(self.strides)

    def _init_layers(self):
        """
        This function builds network layers for category and instance branches.
        It initializes:
          - self.cate_head: nn.ModuleList of intermediate layers for category branch
          - self.ins_head: nn.ModuleList of intermediate layers for instance branch
          - self.cate_out: Output layer for category branch
          - self.ins_out_list: nn.ModuleList of output layers for instance branch, one for each FPN level
        """
        num_groups = 32

        # Category branch head
        self.cate_head = nn.ModuleList()
        for _ in range(self.stacked_convs):
            self.cate_head.append(
                conv_gn_relu(self.seg_feat_channels, self.seg_feat_channels, num_groups=num_groups)
            )

        self.cate_out = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.cate_out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )

        # Instance branch head
        self.ins_head = nn.ModuleList()
        for i in range(self.stacked_convs):
            in_channels = self.seg_feat_channels + 2 if i == 0 else self.seg_feat_channels
            self.ins_head.append(
                conv_gn_relu(in_channels, self.seg_feat_channels, num_groups=num_groups)
            )

        # Instance branch output layers
        self.ins_out_list = nn.ModuleList()
        for num_grid in self.seg_num_grids:
            self.ins_out_list.append(
                nn.Sequential(
                    nn.Conv2d(self.seg_feat_channels, num_grid ** 2, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.Sigmoid(),
                )
            )

    def _init_weights(self):
        """
        This function initializes weights for the head network.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fpn_feat_list, evaluate=False):
        """
        Forward function processes every level in the FPN.
        Input:
        - fpn_feat_list: List of FPN features
        Output:
        - cate_pred_list: List of category predictions
        - ins_pred_list: List of instance predictions
        """
        new_fpn_list = self.NewFPN(fpn_feat_list)  # Adjust FPN features to desired strides
        #print('New FPN feats')
        #print(new_fpn_list)

        upsample_shape = [feat * 2 for feat in new_fpn_list[0].shape[-2:]]  # For evaluation

        cate_pred_list, ins_pred_list = self.MultiApply(
            self.forward_single_level,
            new_fpn_list,
            [*range(len(new_fpn_list)),],
            evaluate=evaluate,
            upsample_shape=upsample_shape
        )

        # Check flag
        assert len(new_fpn_list) == len(self.seg_num_grids)
        assert ins_pred_list[1].shape[1] == self.seg_num_grids[1] ** 2
        assert cate_pred_list[1].shape[2] == self.seg_num_grids[1]

        return cate_pred_list, ins_pred_list

    def NewFPN(self, fpn_feat_list):
        """
        Adjust the original FPN feature maps to have strides [8,8,16,32,32].
        The sizes of the feature maps are adjusted by interpolation.
        """
        # Adjust level 0 and level 4 feature maps
        fpn_p2 = F.interpolate(fpn_feat_list[0], size=fpn_feat_list[1].shape[-2:], mode="bilinear", align_corners=False)
        fpn_p5 = F.interpolate(fpn_feat_list[4], size=fpn_feat_list[3].shape[-2:], mode="bilinear", align_corners=False)
        new_fpn = [fpn_p2, fpn_feat_list[1], fpn_feat_list[2], fpn_feat_list[3], fpn_p5]
        return new_fpn

    def forward_single_level(self, fpn_feat, idx, evaluate=False, upsample_shape=None):
        """
        This function forwards a single level of FPN feature map through the network.
        Input:
        - fpn_feat: (batch_size, fpn_channels, H_feat, W_feat)
        - idx: Index of the FPN level
        Output:
        - cate_pred: Category prediction
        - ins_pred: Instance prediction
        """
        #print('Forward Single Level', idx)
        #print('FPN feat')
        #print(fpn_feat)

        num_grid = self.seg_num_grids[idx]
        batch_size = fpn_feat.shape[0]

        # Category branch
        #cate_feat = F.interpolate(fpn_feat, size=(num_grid, num_grid), mode="bilinear", align_corners=False)
        cate_feat = F.interpolate(fpn_feat, size=(num_grid, num_grid))#, mode="bilinear", align_corners=False)
        #print('Cate', cate_feat.shape)
        #print(cate_feat)

        for conv in self.cate_head:
            cate_feat = conv(cate_feat)
        cate_pred = self.cate_out(cate_feat)  # (batch_size, C, S, S)

        
        
        #print(fpn_feat.shape)
        #print(cate_pred.shape)

        # Instance branch
        # Generate coordinate feature
        coord_feat = self.generate_coordinate(fpn_feat.shape[2:], fpn_feat.device)
        coord_feat = coord_feat.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        ins_feat = torch.cat([fpn_feat, coord_feat], dim=1)
        for conv in self.ins_head:
            ins_feat = conv(ins_feat)

        # Upsample the ins_feat by factor 2
        ins_feat = F.interpolate(ins_feat, scale_factor=2, mode="bilinear", align_corners=True)

        #print('Ins pred')
        #print(ins_feat)

        # Apply the output layer
        ins_pred = self.ins_out_list[idx](ins_feat)  # (batch_size, S^2, 2H_feat, 2W_feat)

        if evaluate:
            # Upsample ins_pred to upsample_shape
            #ins_pred = F.interpolate(ins_pred, size=upsample_shape, mode="bilinear", align_corners=False)
            ins_pred = F.interpolate(ins_pred, size=upsample_shape)
            # Apply points NMS to cate_pred
            cate_pred = self.points_nms(cate_pred).permute(0, 2, 3, 1)  # (batch_size, S, S, C)

        # Check flag
        if not evaluate:
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (num_grid ** 2, fpn_feat.shape[2] * 2, fpn_feat.shape[3] * 2)
        else:
            pass

        return cate_pred, ins_pred

    def points_nms(self, heat, kernel=2):
        """
        This function applies NMS on the heat map (cate_pred), grid-level.
        Input:
        - heat: (batch_size, C-1, S, S)
        Output:
        - Heat after NMS
        """
        hmax = F.max_pool2d(heat, kernel_size=kernel, stride=1, padding=1)
        keep = (hmax[:, :, :-1, :-1] == heat).float()
        return heat * keep

    def generate_coordinate(self, shape, device):
        """
        Generate coordinate feature map.
        """
        x_range = torch.linspace(0, 1, shape[1], device=device)
        y_range = torch.linspace(0, 1, shape[0], device=device)
        y, x = torch.meshgrid(y_range, x_range)
        return torch.stack([x, y], dim=0)  # (2, H, W)

    def MultiApply(self, func, *args, **kwargs):
        """
        Apply function to a list of arguments.
        """
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)
        return (*zip(*map_results),)

    def target(self, ins_pred_list, bbox_list, label_list, mask_list):
        """
        Build the ground truth tensor for each batch in the training.
        Input:
        - bbox_list: List of bounding boxes for each image in the batch
        - label_list: List of labels for each image in the batch
        - mask_list: List of masks for each image in the batch
        Output:
        - ins_gts_list: List of instance ground truths
        - ins_ind_gts_list: List of instance indices
        - cate_gts_list: List of category ground truths
        """
        featmap_sizes = [[pred.shape[2], pred.shape[3]] for pred in ins_pred_list]
        featmap_sizes = [featmap_sizes for _ in range(len(mask_list))]

        output = map(self.target_single_img, bbox_list, label_list, mask_list, featmap_sizes)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = zip(*output)

        # Check flag
        assert ins_gts_list[0][1].shape == (self.seg_num_grids[1] ** 2, 200, 272)
        assert ins_ind_gts_list[0][1].shape == (self.seg_num_grids[1] ** 2,)
        assert cate_gts_list[0][1].shape == (self.seg_num_grids[1], self.seg_num_grids[1])

        return ins_gts_list, ins_ind_gts_list, cate_gts_list

    def target_single_img(self, bounding_boxes, labels, masks, featmap_sizes):
        """
        Process single image to generate target labels for each feature pyramid level.
        Input:
        - bounding_boxes (tensor): Shape (n_obj, 4) in x1y1x2y2 format
        - labels (tensor): Shape (n_obj,)
        - masks (tensor): Shape (n_obj, H_ori, W_ori)
        - featmap_sizes (list): Sizes of feature maps for each level
        Output:
        - tuple: Lists of instance labels, instance indices, and category labels
        """
        # Ensure tensors are on the same device
        device = bounding_boxes.device
        masks = masks.to(device)
        labels = labels.to(device)

        # Extract image height and width from mask dimensions
        img_h, img_w = masks.shape[2], masks.shape[3]

        # Compute bounding box areas and regions (center of mass + normalized dimensions)
        box_areas = torch.sqrt((bounding_boxes[:, 2] - bounding_boxes[:, 0]) * 
                            (bounding_boxes[:, 3] - bounding_boxes[:, 1]))
        regions = torch.zeros((masks.shape[0], 4), device=device)  # Make sure regions is on the same device

        # Compute center of mass for each mask
        for i in range(masks.shape[0]):
            center_y, center_x = center_of_mass(masks[i, 0, :, :].detach().cpu().numpy())  # (y, x order from numpy)
            regions[i, 0] = center_x  # center x
            regions[i, 1] = center_y  # center y

        # Normalize region dimensions
        regions[:, 2] = (bounding_boxes[:, 2] - bounding_boxes[:, 0]) * 0.2 / img_w  # width
        regions[:, 3] = (bounding_boxes[:, 3] - bounding_boxes[:, 1]) * 0.2 / img_h  # height
        regions[:, 0] /= img_w  # Normalize center x by image width
        regions[:, 1] /= img_h  # Normalize center y by image height

        # Lists to store labels for each feature map level
        instance_labels = []
        instance_idx_labels = []
        category_labels = []

        # Process for each feature map size
        for i, feat_size in enumerate(featmap_sizes):
            # Determine scale-based index mask
            if i == 0:
                idx = box_areas < 96
            elif i == 1:
                idx = torch.logical_and(box_areas < 192, box_areas > 48)
            elif i == 2:
                idx = torch.logical_and(box_areas < 384, box_areas > 96)
            elif i == 3:
                idx = torch.logical_and(box_areas < 768, box_areas > 192)
            else:  # i == 4
                idx = box_areas >= 384

            # If no objects in this scale range, create empty label maps
            if torch.sum(idx) == 0:
                empty_cat_label = torch.zeros((self.seg_num_grids[i], self.seg_num_grids[i]), device=device)
                empty_ins_label = torch.zeros((self.seg_num_grids[i]**2, feat_size[0], feat_size[1]), device=device)
                empty_ins_idx_label = torch.zeros(self.seg_num_grids[i]**2, dtype=torch.bool, device=device)

                category_labels.append(empty_cat_label)
                instance_labels.append(empty_ins_label)
                instance_idx_labels.append(empty_ins_idx_label)
                continue

            # Extract relevant regions for selected objects
            selected_regions = regions[idx, :]

            # Compute grid indices for region boundaries
            left_idx = ((selected_regions[:, 0] - selected_regions[:, 2] / 2) * self.seg_num_grids[i]).int()
            right_idx = ((selected_regions[:, 0] + selected_regions[:, 2] / 2) * self.seg_num_grids[i]).int()
            top_idx = ((selected_regions[:, 1] - selected_regions[:, 3] / 2) * self.seg_num_grids[i]).int()
            bottom_idx = ((selected_regions[:, 1] + selected_regions[:, 3] / 2) * self.seg_num_grids[i]).int()

            # Clip indices to grid boundaries
            left = torch.clamp(left_idx, min=0)
            right = torch.clamp(right_idx, max=self.seg_num_grids[i] - 1)
            top = torch.clamp(top_idx, min=0)
            bottom = torch.clamp(bottom_idx, max=self.seg_num_grids[i] - 1)

            # Adjust region centers to grid space
            center_x = (selected_regions[:, 0] * self.seg_num_grids[i]).int()
            center_y = (selected_regions[:, 1] * self.seg_num_grids[i]).int()

            # Define label grids
            cat_label_grid = torch.zeros((self.seg_num_grids[i], self.seg_num_grids[i]), device=device)
            ins_label_grid = torch.zeros((self.seg_num_grids[i]**2, feat_size[0], feat_size[1]), device=device)
            ins_idx_label = torch.zeros(self.seg_num_grids[i]**2, dtype=torch.bool, device=device)

            # Interpolate masks to the current feature map size
            mask_interpolated = torch.nn.functional.interpolate(masks[idx, :, :, :], size=(feat_size[0], feat_size[1]))
            mask_interpolated = (mask_interpolated > 0.5).float()  # Binarize the masks

            # Assign labels to grid cells
            for j in range(center_x.size(0)):
                # Assign category labels to the corresponding grid cells
                cat_label_grid[top[j]:bottom[j] + 1, left[j]:right[j] + 1] = labels[idx][j]

                # Mark positive grid cells for mask assignment
                mask_flag = torch.zeros_like(cat_label_grid)
                mask_flag[top[j]:bottom[j] + 1, left[j]:right[j] + 1] = 1

                positive_idx = torch.flatten(mask_flag) > 0
                ins_label_grid[positive_idx, :, :] = mask_interpolated[j, 0, :, :]

                # Update instance index label for positive cells
                ins_idx_label = torch.logical_or(ins_idx_label, positive_idx)

            # Append to output lists
            category_labels.append(cat_label_grid)
            instance_labels.append(ins_label_grid)
            instance_idx_labels.append(ins_idx_label)

        return instance_labels, instance_idx_labels, category_labels


    def loss(self, cate_pred_list, ins_pred_list, ins_gts_list, ins_ind_gts_list, cate_gts_list):
        """
        Compute loss for a batch of images.
        """
        ins_gts = [
            torch.cat([
                ins_labels_level_img[ins_ind_labels_level_img, ...]
                for ins_labels_level_img, ins_ind_labels_level_img in zip(ins_labels_level, ins_ind_labels_level)
            ], dim=0)
            for ins_labels_level, ins_ind_labels_level in zip(zip(*ins_gts_list), zip(*ins_ind_gts_list))
        ]
        ins_preds = [
            torch.cat([
                ins_preds_level_img[ins_ind_labels_level_img, ...]
                for ins_preds_level_img, ins_ind_labels_level_img in zip(ins_preds_level, ins_ind_labels_level)
            ], dim=0)
            for ins_preds_level, ins_ind_labels_level in zip(ins_pred_list, zip(*ins_ind_gts_list))
        ]

        cate_gts = torch.cat([
            torch.cat([
                cate_gts_level_img.flatten()
                for cate_gts_level_img in cate_gts_level
            ], dim=0)
            for cate_gts_level in zip(*cate_gts_list)
        ], dim=0).to(torch.long)
        cate_preds = torch.cat([
            cate_pred_level.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred_level in cate_pred_list
        ], dim=0)

        lc = self.FocalLoss(cate_preds, cate_gts)
        lm = torch.cat([
            self.DiceLoss(ins_pred, ins_gt)
            for ins_pred, ins_gt in zip(ins_preds, ins_gts)
        ], dim=0).mean()
        return lc * self.cate_loss_cfg["weight"] + lm * self.mask_loss_cfg["weight"], lc, lm

    def DiceLoss(
        self,
        mask_pred: torch.Tensor,    # (..., 2H_feat, 2W_feat)
        mask_gt: torch.Tensor       # (..., 2H_feat, 2W_feat)
    ):
        """
        Compute the Dice Loss.
        """
        return torch.sum((mask_pred - mask_gt) ** 2, dim=[-2, -1]) / torch.sum(mask_pred ** 2 + mask_gt ** 2, dim=[-2, -1])

    def FocalLoss(
        self,
        cate_preds: torch.Tensor,   # (bsz * fpn * S^2, C - 1)
        cate_gts                    # (bsz * fpn * S^2)
    ):
        """
        Compute the Focal Loss.
        """
        indices = cate_gts > 0
        mask = torch.full_like(cate_preds, False, dtype=bool)
        mask[indices] = F.one_hot(cate_gts[indices] - 1, num_classes=self.cate_out_channels).to(torch.bool)
        
        a = torch.where(mask, self.cate_loss_cfg["alpha"], 1 - self.cate_loss_cfg["alpha"])
        p = torch.where(mask, cate_preds, 1 - cate_preds).clamp(min=1e-9)
        return -torch.mean(a * torch.log(p) * (1 - p) ** self.cate_loss_cfg["gamma"])

    def PostProcess(
        self,
        ins_pred_list: List[torch.Tensor],  # fpn x (bsz, S^2, ori_H / 4, ori_W / 4)
        cate_pred_list: List[torch.Tensor], # fpn x (bsz, S, S, C - 1)
        ori_size: Tuple[int, int]           # (ori_H, ori_W)
    ):
        """
        Post-process the predictions.
        """
        return (*map(lambda l: torch.stack(l, dim=0), zip(*[
            self.PostProcessImg(ins_pred_img, cate_pred_img, ori_size)
            for ins_pred_img, cate_pred_img in zip(
                torch.cat(ins_pred_list, dim=1),
                torch.cat([t.flatten(1, 2) for t in cate_pred_list], dim=1),
            )
        ])),)

    def PostProcessImg(
        self,
        ins_pred_img: torch.Tensor,         # (sum_S^2, ori_H / 4, ori_W / 4)
        cate_pred_img: torch.Tensor,        # (sum_S^2, C - 1)
        ori_size: Tuple[int, int]
    ) -> Tuple[
        torch.Tensor,                       # (keep_instance,)
        torch.Tensor,                       # (keep_instance,)
        torch.Tensor,                       # (keep_instance, ori_H, ori_W)
    ]:
        """
        Post-process predictions for a single image.
        """
        # indices = torch.argmax(cate_pred_img, dim=-1) > 0
        # ins_pred_img, cate_pred_img = ins_pred_img[indices], cate_pred_img[indices]

        # scores, labels = torch.max(cate_pred_img, dim=-1)

        # indices, = torch.where(scores > self.postprocess_cfg["cate_thresh"])
        # scores, idx = torch.sort(scores[indices], descending=True)
        # indices = indices[idx]

        # ins_pred_img, labels = ins_pred_img[indices], labels[indices]
        # ins_pred_img = torchvision.transforms.Resize(ori_size)(ins_pred_img)

        # decayed_scores = self.MatrixNMS(ins_pred_img, scores)
        # decayed_scores, idx = torch.topk(decayed_scores, self.postprocess_cfg["keep_instance"])

        # ins_pred_img, scores, labels = ins_pred_img[idx], scores[idx], labels[idx]
        # ins_pred_img = (ins_pred_img > self.postprocess_cfg["ins_thresh"]).to(torch.float)

        # return scores, labels, ins_pred_img

        scores, labels = torch.max(cate_pred_img, dim=-1)

        indices, = torch.where(scores > self.postprocess_cfg["cate_thresh"])
        scores, idx = torch.sort(scores[indices], descending=True)
        indices = indices[idx]

        ins_pred_img, labels = ins_pred_img[indices], labels[indices]
        ins_pred_img = torchvision.transforms.Resize(ori_size)(ins_pred_img)

        decayed_scores = self.MatrixNMS(ins_pred_img, scores)
        decayed_scores, idx = torch.topk(decayed_scores, self.postprocess_cfg["keep_instance"])

        ins_pred_img, scores, labels = ins_pred_img[idx], scores[idx], labels[idx]
        ins_pred_img = (ins_pred_img > self.postprocess_cfg["ins_thresh"]).to(torch.float)
        return scores, labels + 1, ins_pred_img

    def MatrixNMS(self, sorted_masks, sorted_scores, method="gauss", gauss_sigma=0.5):
        """
        Perform Matrix NMS.
        """
        n = len(sorted_scores)
        sorted_masks = sorted_masks.reshape(n, -1)
        intersection = torch.mm(sorted_masks, sorted_masks.T)
        areas = torch.sum(sorted_masks, dim=1, keepdim=True)
        union = areas + areas.T - intersection
        ious = (intersection / union).triu(diagonal=1)

        ious_cmax = ious.max(dim=0)[0].expand(n, n).T
        if method == "gauss":
            decay = torch.exp(-(ious ** 2 - ious_cmax ** 2) / gauss_sigma)
        else:
            decay = (1 - ious) / (1 - ious_cmax)
        decay = decay.min(dim=0)[0]
        return sorted_scores * decay

    def PlotGT(self, ins_gts_list, ins_ind_gts_list, cate_gts_list, color_list, img):
        """
        Visualize the ground truth tensor.
        """
        for img_idx in range(len(img)):
            num_pyramids = len(ins_gts_list[img_idx])

            fig, axes = plt.subplots(1, num_pyramids, figsize=(num_pyramids * 5, 5))
            if num_pyramids == 1:
                axes = [axes]

            for i in range(num_pyramids):
                ax = axes[i]
                _img = img[img_idx].permute(1, 2, 0)
                _img = ((_img - _img.min()) / (_img.max() - _img.min())).numpy(force=True)
                ax.imshow(_img)
                ax.set_title(f"Pyramid level {i + 1}")

                if ins_ind_gts_list[img_idx][i].sum() == 0:
                    continue

                index = ins_ind_gts_list[img_idx][i] > 0
                label = torch.flatten(cate_gts_list[img_idx][i])[index]
                mask = ins_gts_list[img_idx][i][index, :, :]
                mask = mask.unsqueeze(1)
                reshaped_mask = F.interpolate(mask, img.shape[2:], mode="bilinear")
                combined_mask = np.zeros((*img.shape[2:], img.shape[1]))

                for idx, l in enumerate(label):
                    if l > 0:
                        combined_mask[:, :, int(l.item()) - 1] += reshaped_mask[idx, 0, :, :].numpy(force=True)

                origin_img = _img
                index_to_mask = combined_mask > 0
                masked_image = deepcopy(origin_img)
                masked_image[index_to_mask] = 0
                mask_to_plot = combined_mask + masked_image
                ax.imshow(mask_to_plot)

            plt.tight_layout()
            plt.show()

    def PlotInfer(self, NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list, color_list, img, iter_ind):
        """
        Plot inference segmentation results on the image.
        """
        score_threshold = 0.3
        for img_idx in range(len(img)):
            _img = img[img_idx].permute(1, 2, 0)
            _img = ((_img - _img.min()) / (_img.max() - _img.min())).numpy(force=True)
            plt.imshow(_img) ; plt.show()
            plt.title("Inference")


            label = NMS_sorted_cate_label_list[img_idx]
            scores = NMS_sorted_scores_list[img_idx]
            mask = NMS_sorted_ins_list[img_idx].unsqueeze(1)

            combined_mask = np.zeros((*img.shape[2:], img.shape[1]))
            for idx, (l, s) in enumerate(zip(label, scores)):
                if l > 0 and s > score_threshold:
                    combined_mask[:, :, l - 1] += mask[idx, 0, :, :].numpy(force=True)

            origin_img = _img
            index_to_mask = combined_mask > 0
            masked_image = deepcopy(origin_img)
            masked_image[index_to_mask] = 0
            mask_to_plot = combined_mask + masked_image
            plt.imshow(mask_to_plot)

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    parent_dir = "./data"
    paths = {
        fname.split("_")[2]: f"{parent_dir}/{fname}"
        for fname in os.listdir(parent_dir)
    }
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 2
    train_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    resnet50_fpn = Resnet50Backbone()
    solo_head = SOLOHead(num_classes=4)  ## class number is 4, because consider the background as one category.
    # loop the image
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):
        img, label_list, mask_list, bbox_list = [data[i] for i in range(len(data))]
        # fpn is a dict
        backout = resnet50_fpn(img)
        fpn_feat_list = list(backout.values())
        # make the target

        ## demo
        cate_pred_list, ins_pred_list = solo_head.forward(fpn_feat_list, eval=False)
        ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(ins_pred_list,
                                                                         bbox_list,
                                                                         label_list,
                                                                         mask_list)
        mask_color_list = ["jet", "ocean", "Spectral"]
        solo_head.PlotGT(ins_gts_list, ins_ind_gts_list, cate_gts_list, mask_color_list, img)
