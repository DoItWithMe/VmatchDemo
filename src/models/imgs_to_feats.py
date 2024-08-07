import numpy as np
import torch.nn as nn
from models.iscnet import ISCNet
from enum import Enum
from torchvision.transforms import Compose
from PIL import Image


class FeaturesType(Enum):
    TRANSVCL = 1


def gen_img_feats_by_ISCNet(
    imgs_path_list: list[str],
    model: nn.Module,
    preprocessor: Compose,
    device: str = "cuda"
):
    """
    Generate image features by ISCNet model

    Args:
        imgs_path_list (`list[str]`):
            Weight file path.
        model (`nn.Module`):
            ISCNet model
        preprocessor (`Compose`):
            Image preprocessing operations must include converting images into torch.Tensor
        device (`str='cuda'`):
            Devices for model inference, must be same as the model use.

    Returns:
        feats_array (`NDArray[numpy[n,m], dtype=np.float32]`):
            the NDArray of imgs features\n
            n = len(imgs_path_list)\n
            m = model output features dim\n

    ISCNet Usage:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"\n
        image = Image.open(requests.get(url, stream=True).raw)\n
        x = preprocessor(image).unsqueeze(0)\n
        y = model(x)\n
        print(y.shape)  # => torch.Size([1, 256])\n
    """
    if not isinstance(model, ISCNet):
        raise RuntimeError(f"unknown model: {type(model)}")

    if preprocessor is None:
        raise RuntimeError(f"processcessor is not set!")

    # feats = np.zeros([len(imgs_path_list), model.get_output_dim()], dtype=np.float32)
    feats_list = list()
    for img_path in imgs_path_list:
        img = Image.open(img_path)
        # print(f"src shape: {preprocessor(img).shape}, dst shape: {preprocessor(img).unsqueeze(0).shape}")
        img_tensor = preprocessor(img).unsqueeze(0).to(device)
        img_feat = model(img_tensor).detach().cpu().numpy()
       
        # print(f"img feat: { img_feat.shape}, {type( img_feat[0][0])}")
        feats_list.append(img_feat.reshape(-1))
    
    feats_array = np.array(feats_list)
    return feats_array
    # print(f"feats_array: {feats_array.shape}")
    

