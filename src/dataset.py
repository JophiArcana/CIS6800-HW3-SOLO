# Author: Lishuo Pan 2020/4/18
import h5py
import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader

from settings import DEVICE


class BuildDataset(Dataset):
    def __init__(self, paths: Dict[str, str]):
        # TODO: load dataset, make mask list
        # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox

        padding = (11, 0)
        self.labels = [*map(torch.tensor, np.load(paths["labels"], allow_pickle=True))]
        def bbox_transform(arr: np.ndarray) -> torch.Tensor:
            scale = 8.0 / 3.0
            return (torch.tensor(arr).view(-1, 2, 2) * scale + torch.tensor(padding)).view(-1, 4)
        self.transformed_bboxes = [*map(bbox_transform, np.load(paths["bboxes"], allow_pickle=True))]
        cumulative_lengths = np.cumsum((0, *map(len, self.labels)))

        with h5py.File(paths["img"], "r") as f:
            self.img_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(800, 1066), antialias=False),
                torchvision.transforms.Normalize(
                    mean=torch.tensor([0.485, 0.456, 0.406]),
                    std=torch.tensor([0.229, 0.224, 0.225]),
                ),
                torchvision.transforms.Pad(padding=padding),
            ])
            self.img = torch.tensor(np.array(f["data"])).to(torch.float)

        with h5py.File(paths["mask"], "r") as f:
            mask_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(800, 1066), antialias=False),
                torchvision.transforms.Pad(padding=(11, 0)),
            ])
            transformed_mask = mask_transform(torch.tensor(np.array(f["data"])).to(torch.float))
            self.transformed_mask = [
                transformed_mask[cumulative_lengths[i]:cumulative_lengths[i + 1]].unsqueeze(1)
                for i in range(len(self.img))
            ]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: __getitem__
        transformed_img = self.img_transform(self.img[index])
        transformed_masks = self.transformed_mask[index]
        labels = self.labels[index]
        transformed_bboxes = self.transformed_bboxes[index]

        # check flag
        assert transformed_img.shape == (3, 800, 1088)
        assert transformed_bboxes.shape[0] == transformed_masks.shape[0]
        return transformed_img, labels, transformed_masks, transformed_bboxes

    def __len__(self):
        return len(self.img)


class BuildDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, **kwargs):
        def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[
            torch.Tensor, torch.Tensor, Sequence[torch.Tensor], Sequence[torch.Tensor]
        ]:
            img, labels, masks, bboxes = zip(*batch)
            return torch.stack(img, dim=0), labels, masks, bboxes

        super().__init__(
            dataset=dataset,
            collate_fn=collate_fn,
            **{k: v for k, v in kwargs.items() if k != "collate_fn"},
        )


# Visualize debugging
if __name__ == "__main__":
    # file path and make a list
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
    # set 80% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    batch_size = 2
    train_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # loop the image
    for iter, data in enumerate(train_loader):
        img, label, mask, bbox = data
        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size
    
        # plot the origin img
        for i in range(batch_size):
            # TODO: plot images with annotations
    
            _img = img[i].permute(1, 2, 0)
            plt.imshow((_img - _img.flatten(0, 1).min(dim=0).values) / (_img.flatten(0, 1).max(dim=0).values - _img.flatten(0, 1).min(dim=0).values))
            for _label, _mask, _bbox in zip(label[i], mask[i], bbox[i]):
                plt.plot(
                    [_bbox[0], _bbox[0], _bbox[2], _bbox[2], _bbox[0]],
                    [_bbox[1], _bbox[3], _bbox[3], _bbox[1], _bbox[1]],
                    'r', linewidth=2
                )
                
                plt.imshow(_mask[0, :, :], cmap=mask_color_list[_label], alpha=0.5 * _mask[0, :, :])
    
            #plt.savefig(f"./testfig/visualtrainset{iter}.png")
            plt.show()
    
        if iter == 10:
            break
