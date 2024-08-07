import sys
import os

_project_dipath: str = os.path.dirname((os.path.abspath(__file__)))
sys.path.append(_project_dipath)


import argparse
import torch
import numpy as np

from models.iscnet import create_isc_model
from models.imgs_to_feats import gen_img_feats_by_ISCNet
from ffmpeg_utils import generate_1fps_imgs

DEVICE_LIST = ["cpu", "cuda"]


def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ffmpeg",
        type=str,
        help="ffmpeg bin file path",
        default="./assets/ffmpeg",
        required=False,
    )
    parser.add_argument(
        "--isc-weight",
        type=str,
        help="isc weight path",
        default="./assets/models/isc_ft_v107.pth.tar",
        required=False,
    )
    parser.add_argument(
        "--transVCS-weight",
        type=str,
        help="transVCS   path",
        default="./assets/models/tarnsVCL_model_1.pth",
        required=False,
    )

    parser.add_argument("--device", type=str, help="cpu or cuda", default="cuda")

    parser.add_argument(
        "--output-dir",
        type=str,
        help="output dir path, auto-create it if not exisit",
        default="./output",
        required=False,
    )

    parser.add_argument(
        "--sample-file-path",
        "-s",
        type=str,
        help="input sample media file path",
        required=True,
    )
    parser.add_argument(
        "--reference-file-path",
        "-r",
        type=str,
        help="input reference media file path",
        required=True,
    )

    return parser.parse_args()


def check_args(args):
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("gpu is not available, use cpu")
            args.device = "cpu"

    if args.device not in DEVICE_LIST:
        print(f"unkown device: {args.device}, only thess is available: {DEVICE_LIST}")
        exit(-1)


if __name__ == "__main__":
    print("hi")
    args = parser_args()
    check_args(args)

    device = args.device
    reference_file_path = args.reference_file_path
    ffmpeg_path = args.ffmpeg
    sample_file_path = args.sample_file_path
    output_dir = args.output_dir

    isc_weight_path = args.isc_weight

    # get 1fps imgs from reference media file and sample media file
    print(f"start get 1pfs imgs from {reference_file_path}")
    ref_imgs_dir_path = generate_1fps_imgs(ffmpeg_path, reference_file_path, output_dir)
    ref_imgs_list = [
        os.path.join(ref_imgs_dir_path, img_name)
        for img_name in os.listdir(ref_imgs_dir_path)
    ]

    print(f"start get 1pfs imgs from {args.sample_file_path}")
    sample_imgs_dir_path = generate_1fps_imgs(ffmpeg_path, sample_file_path, output_dir)

    sample_imgs_list = [
        os.path.join(sample_imgs_dir_path, img_name)
        for img_name in os.listdir(sample_imgs_dir_path)
    ]

    print("create isc model")
    isc_model, isc_processer = create_isc_model(
        weight_file_path=isc_weight_path, device=device, is_training=False
    )

    print("gen sample feats")
    gen_img_feats_by_ISCNet(sample_imgs_list, isc_model, isc_processer, device)

    print("gen ref feats")
    gen_img_feats_by_ISCNet(ref_imgs_list, isc_model, isc_processer, device)
