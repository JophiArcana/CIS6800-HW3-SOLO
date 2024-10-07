from copy import deepcopy
from functools import partial
from types import MappingProxyType

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import center_of_mass

from backbone import *
from dataset import *


def conv_gn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_groups=32, bias=False):
    """Helper function to create a Conv2d -> GroupNorm -> ReLU layer."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.GroupNorm(num_groups, out_channels),
        nn.ReLU(inplace=True)
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
            "keep_instance": 5,
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
            nn.Conv2d(self.seg_feat_channels, self.num_classes - 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
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
                    nn.Sigmoid()
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

    def forward(self, fpn_feat_list, eval=False):
        """
        Forward function processes every level in the FPN.
        Input:
        - fpn_feat_list: List of FPN features
        Output:
        - cate_pred_list: List of category predictions
        - ins_pred_list: List of instance predictions
        """
        new_fpn_list = self.NewFPN(fpn_feat_list)  # Adjust FPN features to desired strides
        upsample_shape = [feat * 2 for feat in new_fpn_list[0].shape[-2:]]  # For evaluation

        cate_pred_list, ins_pred_list = self.MultiApply(
            self.forward_single_level,
            new_fpn_list,
            [*range(len(new_fpn_list)),],
            eval=eval,
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

    def forward_single_level(self, fpn_feat, idx, eval=False, upsample_shape=None):
        """
        This function forwards a single level of FPN feature map through the network.
        Input:
        - fpn_feat: (batch_size, fpn_channels, H_feat, W_feat)
        - idx: Index of the FPN level
        Output:
        - cate_pred: Category prediction
        - ins_pred: Instance prediction
        """
        num_grid = self.seg_num_grids[idx]
        batch_size = fpn_feat.shape[0]

        # Category branch
        cate_feat = F.interpolate(fpn_feat, size=(num_grid, num_grid), mode="bilinear", align_corners=False)
        for conv in self.cate_head:
            cate_feat = conv(cate_feat)
        cate_pred = self.cate_out(cate_feat)  # (batch_size, C-1, S, S)

        # Instance branch
        # Generate coordinate feature
        coord_feat = self.generate_coordinate(fpn_feat.shape[2:], fpn_feat.device)
        coord_feat = coord_feat.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        ins_feat = torch.cat([fpn_feat, coord_feat], dim=1)
        for conv in self.ins_head:
            ins_feat = conv(ins_feat)
        # Upsample the ins_feat by factor 2
        ins_feat = F.interpolate(ins_feat, scale_factor=2, mode="bilinear", align_corners=False)
        # Apply the output layer
        ins_pred = self.ins_out_list[idx](ins_feat)  # (batch_size, S^2, 2H_feat, 2W_feat)

        if eval:
            # Upsample ins_pred to upsample_shape
            ins_pred = F.interpolate(ins_pred, size=upsample_shape, mode="bilinear", align_corners=False)
            # Apply points NMS to cate_pred
            cate_pred = self.points_nms(cate_pred).permute(0, 2, 3, 1)  # (batch_size, S, S, C-1)

        # Check flag
        if eval == False:
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

    def target_single_img(self, bounding_boxes, labels, masks, featmap_sizes=None):
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
        h, w = masks.shape[2], masks.shape[3]

        # Compute object areas and regions
        area = torch.sqrt((bounding_boxes[:, 2] - bounding_boxes[:, 0]) * (bounding_boxes[:, 3] - bounding_boxes[:, 1]))
        region = torch.zeros((masks.shape[0], 4), device=bounding_boxes.device)

        for i in range(masks.shape[0]):
            center = center_of_mass(masks[i, :, :].cpu().numpy())
            region[i, 0] = center[1] / w
            region[i, 1] = center[0] / h

        region[:, 2] = (bounding_boxes[:, 2] - bounding_boxes[:, 0]) * 0.2 / w
        region[:, 3] = (bounding_boxes[:, 3] - bounding_boxes[:, 1]) * 0.2 / h

        ins_label_list, ins_ind_label_list, cate_label_list = [], [], []

        for i, size in enumerate(featmap_sizes):
            # Determine which objects belong to this feature level
            if i == 0:
                idx = area < 96
            elif i == 1:
                idx = (96 > area) & (area > 48)
            elif i == 2:
                idx = (384 > area) & (area > 96)
            elif i == 3:
                idx = (768 > area) & (area > 192)
            else:
                idx = area >= 384

            grid = self.seg_num_grids[i]

            if not idx.any():
                # If no objects in this level, append empty tensors
                cate_label_list.append(torch.zeros((grid, grid), device=bounding_boxes.device))
                ins_label_list.append(torch.zeros((grid ** 2, size[0], size[1]), device=bounding_boxes.device))
                ins_ind_label_list.append(torch.zeros(grid ** 2, dtype=torch.bool, device=bounding_boxes.device))
                continue

            # Compute grid indices for objects in this level
            region_idx = region[idx, :]
            left_ind, right_ind = ((region_idx[:, 0] - region_idx[:, 2] / 2) * grid).int(), (
                (region_idx[:, 0] + region_idx[:, 2] / 2) * grid).int()
            top_ind, bottom_ind = ((region_idx[:, 1] - region_idx[:, 3] / 2) * grid).int(), (
                (region_idx[:, 1] + region_idx[:, 3] / 2) * grid).int()

            left = torch.clamp(left_ind, 0, grid - 1)
            right = torch.clamp(right_ind, 0, grid - 1)
            top = torch.clamp(top_ind, 0, grid - 1)
            bottom = torch.clamp(bottom_ind, 0, grid - 1)

            xA = torch.clamp((region_idx[:, 0] * grid).int() - 1, left, right)
            xB = torch.clamp((region_idx[:, 0] * grid).int() + 1, left, right)
            yA = torch.clamp((region_idx[:, 1] * grid).int() - 1, top, bottom)
            yB = torch.clamp((region_idx[:, 1] * grid).int() + 1, top, bottom)

            # Initialize tensors for this level
            cat_label = torch.zeros((grid, grid), device=bounding_boxes.device)
            ins_label = torch.zeros((grid ** 2, size[0], size[1]), device=bounding_boxes.device)
            ins_index_label = torch.zeros(grid ** 2, dtype=torch.bool, device=bounding_boxes.device)

            # Interpolate masks to feature map size
            mask_interpolate = F.interpolate(masks[idx, :, :, :], size=(size[0], size[1]), mode="bilinear")
            mask_interpolate = (mask_interpolate > 0.5).float()

            # Assign labels and masks
            for j in range(xA.size(0)):
                cat_label[yA[j]:yB[j] + 1, xA[j]:xB[j] + 1] = labels[idx][j]
                flag_matrix = torch.zeros_like(cat_label)
                flag_matrix[yA[j]:yB[j] + 1, xA[j]:xB[j] + 1] = 1
                positive_index = flag_matrix.flatten() > 0
                ins_label[positive_index, :, :] = mask_interpolate[j, 0, :, :]
                ins_index_label |= positive_index

            cate_label_list.append(cat_label)
            ins_label_list.append(ins_label)
            ins_ind_label_list.append(ins_index_label)

        # Check flag
        assert ins_label_list[1].shape == (1296, 200, 272)
        assert ins_ind_label_list[1].shape == (1296,)
        assert cate_label_list[1].shape == (36, 36)

        return ins_label_list, ins_ind_label_list, cate_label_list

    # The following methods are placeholders for loss computation and post-processing.
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
        lm = self.DiceLoss(torch.cat(ins_preds, dim=0), torch.cat(ins_gts, dim=0)).mean()
        return lc * self.cate_loss_cfg["weight"] + lm * self.mask_loss_cfg["weight"]

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
        cate_preds = torch.cat([cate_preds, 1 - torch.sum(cate_preds, dim=1, keepdim=True)], dim=1)
        mask = F.one_hot(cate_gts, num_classes=self.num_classes).to(torch.bool)
        a = torch.where(mask, self.cate_loss_cfg["alpha"], 1 - self.cate_loss_cfg["alpha"])
        torch.set_printoptions(precision=12, sci_mode=False)
        print(cate_preds)
        raise Exception()
        p = torch.clamp_min(torch.where(mask, cate_preds, 1 - cate_preds), min=1e-6)
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
        print(ins_pred_img.shape, cate_pred_img.shape)
        scores, labels = torch.max(cate_pred_img, dim=-1)

        indices, = torch.where(scores > self.postprocess_cfg["cate_thresh"])
        scores, idx = torch.sort(scores[indices], descending=True)
        indices = indices[idx]

        ins_pred_img, labels = ins_pred_img[indices], labels[indices]
        ins_pred_img = torchvision.transforms.Resize(ori_size)(ins_pred_img)
        print(ins_pred_img.shape, scores.shape, labels.shape)

        decayed_scores = self.MatrixNMS(ins_pred_img, scores)
        decayed_scores, idx = torch.topk(decayed_scores, self.postprocess_cfg["keep_instance"])
        indices = indices[idx]

        ins_pred_img, scores, labels = ins_pred_img[idx], scores[idx], labels[idx]
        ins_pred_img = (ins_pred_img > self.postprocess_cfg["ins_thresh"]).to(torch.float)
        print(scores, labels, ins_pred_img.shape, torch.sum(ins_pred_img, dim=[-2, -1]))
        return scores, labels, ins_pred_img

    def MatrixNMS(self, sorted_masks, sorted_scores, method="gauss", gauss_sigma=0.5):
        """
        Perform Matrix NMS.
        """
        n = len(sorted_scores)
        sorted_masks = sorted_masks.reshape(n, -1)
        intersection = torch.mm(sorted_masks, sorted_masks.T)
        areas = sorted_masks.sum(dim=1).expand(n, n)
        union = areas + areas.T - intersection
        ious = (intersection / union).triu(diagonal=1)

        ious_cmax = ious.max(0)[0].expand(n, n).T
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
        num_pyramids = len(ins_gts_list[0])

        fig, axes = plt.subplots(1, num_pyramids, figsize=(num_pyramids * 5, 5))
        if num_pyramids == 1:
            axes = [axes]

        for i in range(num_pyramids):
            ax = axes[i]
            _img = img[0].permute(1, 2, 0)
            _img = (_img - _img.min()) / (_img.max() - _img.min())
            _img = torch.clamp(_img, 0, 1).cpu().numpy()
            ax.imshow(_img)
            ax.set_title(f"Pyramid level {i + 1}")

            if ins_ind_gts_list[0][i].sum() == 0:
                continue

            index = ins_ind_gts_list[0][i] > 0
            label = torch.flatten(cate_gts_list[0][i])[index]
            mask = ins_gts_list[0][i][index, :, :]
            mask = mask.unsqueeze(1)
            reshaped_mask = F.interpolate(mask, (img.shape[2], img.shape[3]), mode="bilinear")
            combined_mask = np.zeros((img.shape[2], img.shape[3], img.shape[1]))

            for idx, l in enumerate(label):
                l = l.item()
                if l == 1:
                    combined_mask[:, :, 0] += reshaped_mask[idx, 0, :, :].cpu().numpy()
                elif l == 2:
                    combined_mask[:, :, 1] += reshaped_mask[idx, 0, :, :].cpu().numpy()
                elif l == 3:
                    combined_mask[:, :, 2] += reshaped_mask[idx, 0, :, :].cpu().numpy()

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
        pass


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
