import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from dataset import *
from backbone import *
from solo_model import SOLO


if __name__ == "__main__":
    # file path and make a list
    parent_dir = "./data"
    paths = {
        fname.split("_")[2]: f"{parent_dir}/{fname}"
        for fname in os.listdir(parent_dir)
    }

    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator(DEVICE))
    # push the randomized training data into the dataloader

    batch_size = 3
    train_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]

    # images:         (batch_size, 3, 800, 1088)
    # labels:         list with len: batch_size, each (n_obj,)
    # masks:          list with len: batch_size, each (n_obj, 1, 800, 1088), Added channels dimention to work with torchvision functions
    # bounding_boxes: list with len: batch_size, each (n_obj, 4)

    model = SOLO()
    logger = TensorBoardLogger("tensorboard", name="solo")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="solo_{epoch:02d}",
        every_n_epochs=4,
        save_top_k=-1,
    )
    trainer = pl.Trainer(
        max_epochs=36,
        logger=logger,
        callbacks=[checkpoint_callback],
        strategy=DDPStrategy(find_unused_parameters=True, static_graph=True),
    )

    with torch.autograd.set_detect_anomaly(True):
        trainer.fit(model, train_loader, test_loader)

    # check = 0
    # evaluate = False
    # with torch.set_grad_enabled(True):
    #     for iter, data in enumerate(train_loader, 0):   # list of batch_size, mask: 1x3x800x1088, bbox: 1x4, target:
    #         img, label_list, mask_list, bbox_list = data
    #         for label in label_list:
    #             cate_pred_list, ins_pred_list = model.forward(img, evaluate=evaluate)
    #             ins_gts_list, ins_ind_gts_list, cate_gts_list = model.head.target(ins_pred_list, bbox_list, label_list, mask_list)
    #
    #             if not evaluate:
    #                 print(model.training_step(data, iter))
    #             # raise Exception()
    #             # print([ins_pred.shape for ins_pred in ins_pred_list])
    #             # print([cate_pred.shape for cate_pred in cate_pred_list])
    #             else:
    #                 cate_pred_list, ins_pred_list = model.forward(img, evaluate=evaluate)
    #                 print([t.shape for t in model.head.PostProcess(ins_pred_list, cate_pred_list, (800, 1088))])
    #             raise Exception()
    #
    #             check += 1
    #             if check == 10:
    #               break
    #         if check == 10:
    #             break





