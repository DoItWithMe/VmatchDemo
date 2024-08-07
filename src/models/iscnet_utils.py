import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from iscnet.isc_model import ISCNet
from .iscnet.isc_model import ISCNet
from torchvision import transforms
from PIL import Image


def create_isc_model(
    weight_file_path: str,
    fc_dim: int = 256,
    p: float = 1.0,
    eval_p: float = 1.0,
    l2_normalize: bool = True,
    device: str = "cuda",
    is_training: bool = False,
):
    """
    Create a model for image copy-detection task.

    Args:
        weight_file_path (`str=None`):
            Weight file path.
        fc_dim (`int=256`):
            Feature dimension of the fc layer.
        p (`float=1.0`):
            Power used in gem pooling for training.
        eval_p (`float=1.0`):
            Power used in gem pooling for evaluation.
        l2_normalize (`bool=True`):
            Whether to normalize the feature vector.
        device (`str='cuda'`):
            Device to load the model.
        is_training (`bool=False`):
            Whether to load the model for training.

    Returns:
        model:
            ISCNet model.
        preprocessor:
            Preprocess function tied to model.
    """
    ckpt = torch.load(weight_file_path)

    arch = ckpt["arch"]  # tf_efficientnetv2_m_in21ft1k
    input_size = ckpt["args"].input_size

    if arch == "tf_efficientnetv2_m_in21ft1k":
        arch = "timm/tf_efficientnetv2_m.in21k_ft_in1k"

    backbone = timm.create_model(arch, features_only=True)
    model = ISCNet(
        backbone=backbone,
        fc_dim=fc_dim,
        p=p,
        eval_p=eval_p,
        l2_normalize=l2_normalize,
    )

    model.to(device).train(is_training)

    state_dict = {}
    for s in ckpt["state_dict"]:
        state_dict[s.replace("module.", "")] = ckpt["state_dict"][s]

    if fc_dim != 256:
        # interpolate to new fc_dim
        state_dict["fc.weight"] = (
            F.interpolate(
                state_dict["fc.weight"].permute(1, 0).unsqueeze(0),
                size=fc_dim,
                mode="linear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 0)
        )
        for bn_param in ["bn.weight", "bn.bias", "bn.running_mean", "bn.running_var"]:
            state_dict[bn_param] = (
                F.interpolate(
                    state_dict[bn_param].unsqueeze(0).unsqueeze(0),
                    size=fc_dim,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )

    model.load_state_dict(state_dict)

    preprocessor = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=backbone.default_cfg["mean"],
                std=backbone.default_cfg["std"],
            ),
        ]
    )

    if device == "cuda":
        model = torch.nn.DataParallel(model.cuda())

    return model, preprocessor


def gen_img_feats_by_ISCNet(
    imgs_path_list: list[str],
    model: nn.Module,
    preprocessor: transforms.Compose,
    device: str = "cuda",
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
    if isinstance(model, nn.DataParallel):
        if not isinstance(model.module, ISCNet):
            raise RuntimeError(f"unknown model: {type(model)} -- {type(model.module)}")
    else:
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
