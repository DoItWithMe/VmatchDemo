import torch
import torchvision
import torch.nn as nn
from typing import Any
from .transvcl.yolo_pafpn import YOLOPAFPN
from .transvcl.yolo_head import YOLOXHead
from .transvcl.transvcl_model import TransVCL
from collections import defaultdict


def _postprocess(
    prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False
):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections  # type: ignore
        else:
            output[i] = torch.cat((output[i], detections))  # type: ignore

    return output


def create_transvcl_model(
    weight_file_path: str,
    depth: float = 0.33,
    width: float = 0.50,
    act: str = "silu",
    num_classes: int = 1,
    vta_config: dict[str, Any] = dict(),
    device: str = "cuda",
    is_training: bool = False,
):
    """
    Create a transvcl model which basied on YOLO for image copy-detection task.

    Args:
        weight_file_path (`str=None`):
            Weight file path.
        depth (`float=0.33`):
            the depth of network.
        width (`float=0.50`):
            the width of network.
        act (`str="silu"`):
            Types of activation functions.
        num_classes (`bool=True`):
            The number of categories in the classification task.
        vta_config (`dict[str, Any]`):
            some args used by TransVCL, see class TransVCL for the details.
        device (`str='cuda'`):
            Device to load the model.
        is_training (`bool=False`):
            Whether to load the model for training.

    Returns:
        model:
            TransVCL model.
    """
    if len(vta_config) == 0:
        vta_config = {
            "d_model": 256,
            "nhead": 8,
            "layer_names": ["self", "cross"] * 1,
            "attention": "linear",
            "match_type": "dual_softmax",
            "dsmax_temperature": 0.1,
            "keep_ratio": False,
            "unsupervised_weight": 0.5,
        }

    def init_transvcl(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
    head = YOLOXHead(num_classes, width, in_channels=in_channels, act=act)
    model = TransVCL(vta_config, backbone, head)

    model.apply(init_transvcl)
    model.head.initialize_biases(1e-2)

    model.to(device).train(is_training)

    ckpt = torch.load(weight_file_path)
    model.load_state_dict(ckpt["model"])

    if device == "cuda":
        model = torch.nn.DataParallel(model.cuda())

    return model


def query_transVCL(
    model: nn.Module,
    transvcl_batch_feats: list[Any],
    confthre: float,
    nmsthre: float,
    img_size: tuple[int, int],
    feat_max_length: int,
    device: str = "cuda",
):
    """
    Using TransVCL for video copy positioning

    Args:
        model (`nn.Module`):
            TransVCL model

        transvcl_batch_feats (`list[Any]`):
            transvcl batch features, generate by func:: transform_feats.trans_isc_features_to_transVCL_fromat

        confthre (`float`):
            conf threshold of copied segments

        nmsthre (`float`):
            nms threshold of copied segments

        img_size (`tuple[int, int]`):
            length for copied localization module

        feat_max_length (`str`):
            feature length for TransVCL input

        device (`str='cuda'`):
            Devices for model inference, must be same as the model use.

    """
    if isinstance(model, nn.DataParallel):
        if not isinstance(model.module, TransVCL):
            raise RuntimeError(f"unknown model: {type(model)} -- {type(model.module)}")
    else:
        if not isinstance(model, TransVCL):
            raise RuntimeError(f"unknown model: {type(model)}")

    batch_feat_result = {}
    for idx, batch_feat in enumerate(transvcl_batch_feats):
        print(f"start compare {idx}")
        feat1, feat2, mask1, mask2, img_info, file_name = batch_feat
        feat1, feat2, mask1, mask2 = (
            feat1.to(device),
            feat2.to(device),
            mask1.to(device),
            mask2.to(device),
        )

        print(f"query file: {file_name}")
        print(f"feat1: {feat1.shape}")
        print(f"feat2: {feat2.shape}")

        with torch.no_grad():
            model_outputs = model(feat1, feat2, mask1, mask2, file_name, img_info)
            outputs = _postprocess(
                model_outputs[1],
                1,
                confthre,
                nmsthre,
                class_agnostic=True,
            )

            for idx, output in enumerate(outputs):
                if output is not None:
                    bboxes = output[:, :5].cpu()

                    scale1, scale2 = (
                        img_info[0] / img_size[0],
                        img_info[1] / img_size[1],
                    )
                    bboxes[:, 0:4:2] *= scale2[idx]
                    bboxes[:, 1:4:2] *= scale1[idx]
                    batch_feat_result[file_name[idx]] = bboxes[
                        :, (1, 0, 3, 2, 4)
                    ].tolist()
                else:
                    batch_feat_result[file_name[idx]] = [[]]

    result = defaultdict(list)

    for img_name in batch_feat_result:
        img_file = img_name.split("_")[0]
        # i is sample segment seq
        # j is reference segment seq
        i, j = int(img_name.split("_")[1]), int(img_name.split("_")[2])
        if batch_feat_result[img_name] != [[]]:
            for r in batch_feat_result[img_name]:
                result[img_file].append(
                    [
                        r[0] + i * feat_max_length,
                        r[1] + j * feat_max_length,
                        r[2] + i * feat_max_length,
                        r[3] + j * feat_max_length,
                        r[4],
                    ]
                )
