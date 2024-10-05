import torch
import torchvision

from settings import DEVICE


def Resnet50Backbone(checkpoint_file=None, eval=True):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)

    if eval:
        model.eval()

    model.to(DEVICE)
    resnet50_fpn = model.backbone

    if checkpoint_file:
        resnet50_fpn.load_state_dict(torch.load(checkpoint_file, map_location=DEVICE)['backbone'])

    return resnet50_fpn


if __name__ == '__main__':
    resnet50_fpn = Resnet50Backbone()
    # backbone = Resnet50Backbone('checkpoint680.pth')
    E = torch.ones((2, 3, 800, 1088))
    backout = resnet50_fpn(E)
    print(backout.keys())
    print(backout["0"].shape)
    print(backout["1"].shape)
    print(backout["2"].shape)
    print(backout["3"].shape)
    print(backout["pool"].shape)




