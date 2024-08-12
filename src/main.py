import sys
import os

_project_dipath: str = os.path.dirname((os.path.abspath(__file__)))
sys.path.append(_project_dipath)


import argparse
import torch
import numpy as np

from models.iscnet_utils import create_isc_model, gen_img_feats_by_ISCNet
from models.transvcl_utils import create_transvcl_model, gen_match_segments_by_transVCL
from models.transform_feats import trans_isc_features_to_transVCL_fromat
from ffmpeg.ffmpeg_utils import generate_imgs

from loguru import logger as log

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
        "--conf-thre",
        type=float,
        default=0.1,
        help="transVCL: conf threshold of copied segments    ",
    )

    parser.add_argument(
        "--nms-thre",
        type=float,
        default=0.3,
        help="transVCL: nms threshold of copied segments",
    )

    parser.add_argument(
        "--segment-length",
        type=int,
        default=1200,
        help="transVCL: frames number of each compare segment",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="transVCL: length for copied localization module",
    )

    parser.add_argument(
        "--transVCL-weight",
        type=str,
        help="transVCL weight path",
        default="./assets/models/tarnsVCL_model_1.pth",
        required=False,
    )

    parser.add_argument("--device", type=str, help="cpu or cuda", default="cuda")

    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="output fps when converting video to images",
    )

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
            log.info("gpu is not available, use cpu")
            args.device = "cpu"

    if args.device not in DEVICE_LIST:
        log.info(
            f"unkown device: {args.device}, only thess is available: {DEVICE_LIST}"
        )
        exit(-1)


def init_log():
    log.remove()
    log.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}")

if __name__ == "__main__":
    init_log()
    args = parser_args()
    check_args(args)

    device = args.device

    ffmpeg_path = args.ffmpeg

    ref_file_path = args.reference_file_path
    sample_file_path = args.sample_file_path

    output_dir = args.output_dir

    isc_weight_path = args.isc_weight
    transvcl_weight_path = args.transVCL_weight

    confthre = args.conf_thre
    nmsthre = args.nms_thre
    img_size = (args.img_size, args.img_size)
    segment_length = args.segment_length
    fps = args.fps

    # get 1fps imgs from reference media file and sample media file
    # log.info(f"start get 1fps imgs from {ref_file_path}")
    # ref_imgs_dir_path = generate_imgs(ffmpeg_path, ref_file_path, output_dir, fps)
    # ref_imgs_list = [
    #     os.path.join(ref_imgs_dir_path, img_name)
    #     for img_name in os.listdir(ref_imgs_dir_path)
    # ]

    # log.info(f"start get 1fps imgs from {sample_file_path}")
    # sample_imgs_dir_path = generate_imgs(ffmpeg_path, sample_file_path, output_dir, fps)
    # sample_imgs_list = [
    #     os.path.join(sample_imgs_dir_path, img_name)
    #     for img_name in os.listdir(sample_imgs_dir_path)
    # ]

    # log.info("create isc model")
    # isc_model, isc_processer = create_isc_model(
    #     weight_file_path=isc_weight_path, device=device, is_training=False
    # )

    # log.info("gen ref feats")
    # ref_isc_feats = gen_img_feats_by_ISCNet(
    #     ref_imgs_list, isc_model, isc_processer, device
    # )
    # log.info(f"get ref feats: {ref_isc_feats.shape}")

    # log.info("gen sample feats")
    # sample_isc_feats = gen_img_feats_by_ISCNet(
    #     sample_imgs_list, isc_model, isc_processer, device
    # )
    # log.info(f"get sample feats: {sample_isc_feats.shape}")

    # tmp code
    sample_feats_path = os.path.join(
        "/data/jinzijian/assets/vmatch-videos",
        f"{os.path.splitext(os.path.basename(sample_file_path))[0]}.npy",
    )

    ref_feats_path = os.path.join(
        "/data/jinzijian/assets/vmatch-videos",
        f"{os.path.splitext(os.path.basename(ref_file_path))[0]}.npy",
    )

    # log.info("save sample feats")
    # np.save(sample_feats_path, sample_isc_feats)

    # log.info("save ref feats")
    # np.save(ref_feats_path, ref_isc_feats)

    log.info("create transvcl model")
    transvcl_model = create_transvcl_model(
        weight_file_path=transvcl_weight_path, device=device, is_training=False
    )

    log.info("load isc feats")
    sample_isc_feats = np.load(sample_feats_path)
    ref_isc_feats = np.load(ref_feats_path)

    log.info(
        f"isc feat shape: sample: {sample_isc_feats.shape}, ref: {ref_isc_feats.shape}"
    )

    compare_name = (
        os.path.splitext(os.path.basename(sample_file_path))[0]
        + "-"
        + os.path.splitext(os.path.basename(ref_file_path))[0]
    )

    log.info("trans isc feats to transVCL feature format")
    transvcl_batch_feats = trans_isc_features_to_transVCL_fromat(
        sample_isc_feats, ref_isc_feats, compare_name, segment_length
    )

    log.info("query transVCL")
    matched_segments = gen_match_segments_by_transVCL(
        transvcl_model,
        transvcl_batch_feats,
        confthre,
        nmsthre,
        img_size,
        segment_length,
        1000.0,
        1000.0,
        device,
    )
    for matched_seg_title in matched_segments:
        log.info(f"matched_segments: {matched_segments[matched_seg_title]}")
