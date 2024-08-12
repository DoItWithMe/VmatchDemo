import torch
import numpy as np
from loguru import logger as log

def _feat_paddding(feat: torch.Tensor, axis: int, new_size: int, fill_value: int = 0):
    pad_shape = list(feat.shape)
    pad_shape[axis] = max(0, new_size - pad_shape[axis])
    feat_pad = torch.Tensor(*pad_shape).fill_(fill_value)
    return torch.cat([feat, feat_pad], dim=axis)


def trans_isc_features_to_transVCL_fromat(
    sample_feats: np.ndarray, ref_feats: np.ndarray, file_name: str, segment_length:int
):
    """
    Feature transformer for ISCNet features to TransVCL features format

    Args:
        sample_feats (`NDArray[numpy[n,m], dtype=np.float32]`):
            the NDArray of imgs features\n
            n = len(imgs_path_list)\n
            m = model output features dim\n

        ref_feats (`NDArray[numpy[n,m], dtype=np.float32]`):
            the NDArray of imgs features\n
            n = len(imgs_path_list)\n
            m = model output features dim\n

        file_name (`str`):
            compare task name

        segment_length (`int`)
            frames number of each segment, it's ok if real frame length is lesser than segment_length
    """
    # segment_length = 1200
    sample_list, ref_list = [], []
    i, j = -1, -1

    for i in range(len(sample_feats) // segment_length):
        sample_list.append(sample_feats[i * segment_length : (i + 1) * segment_length])

    # log.info(
    #     f"i: {i}, feat length: {segment_length}, sample_feats len: {len(sample_feats)} {len(sample_feats) // segment_length}, sample_list: {len(sample_list)}"
    # )

    for j in range(len(ref_feats) // segment_length):
        ref_list.append(ref_feats[j * segment_length : (j + 1) * segment_length])

    if len(sample_feats) > (i + 1) * segment_length:
        sample_list.append(sample_feats[(i + 1) * segment_length :])

    if len(ref_feats) > (j + 1) * segment_length:
        ref_list.append(ref_feats[(j + 1) * segment_length :])

    # log.info(
    #     f"i: {i}, feat length: {segment_length}, sample_feats len: {len(sample_feats)} {len(sample_feats) // segment_length}, sample_list: {len(sample_list)} {type(sample_list[0])}"
    # )

    batch_list = []
    for i in range(len(sample_list)):
        for j in range(len(ref_list)):
            sample_mask, ref_mask = np.zeros(segment_length, dtype=bool), np.zeros(
                segment_length, dtype=bool
            )
            sample_mask[: len(sample_list[i])] = True
            ref_mask[: len(ref_list[j])] = True

            sample_feat_padding = _feat_paddding(torch.tensor(sample_list[i]), 0, segment_length)
            ref_feat_padding = _feat_paddding(torch.tensor(ref_list[j]), 0, segment_length)
            img_info = [
                torch.tensor([len(sample_list[i])]),
                torch.tensor([len(ref_list[j])]),
            ]

            file_name_idx = file_name + "_" + str(i) + "_" + str(j)

            batch_list.append(
                (
                    sample_feat_padding,
                    ref_feat_padding,
                    torch.from_numpy(sample_mask),
                    torch.from_numpy(ref_mask),
                    img_info,
                    file_name_idx,
                )
            )

    return batch_list
