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
from utils.time_utils import TimeRecorder

DEVICE_LIST = ["cpu", "cuda"]
MIN_SEGMENT_LENGTH = 100


def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ffmpeg",
        type=str,
        help="ffmpeg bin file path",
        default="./assets/ffmpeg",
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
        "--isc-weight",
        type=str,
        help="isc weight path",
        default="./assets/models/isc_ft_v107.pth.tar",
        required=False,
    )

    parser.add_argument(
        "--transVCL-weight",
        type=str,
        help="transVCL weight path",
        default="./assets/models/tarnsVCL_model_1.pth",
        required=False,
    )

    parser.add_argument(
        "--conf-thre",
        type=float,
        default=0.6,
        help="transVCL: conf threshold of copied segments    ",
    )

    parser.add_argument(
        "--nms-thre",
        type=float,
        default=0.3,
        help="transVCL: nms threshold of copied segments",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="transVCL: length for copied localization module",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="output fps when converting video to images",
    )

    parser.add_argument(
        "--segment-duration",
        type=int,
        default=200,
        help="transVCL: segment duration in milliseconds",
    )

    parser.add_argument(
        "--sample-videos-dir",
        "-s",
        type=str,
        help="input sample videos dir",
        required=True,
    )

    parser.add_argument(
        "--reference-videos-dir",
        "-r",
        type=str,
        help="input reference videos dir",
        required=True,
    )

    return parser.parse_args()


def check_args(args):
    if args.device == "cuda":
        if not torch.cuda.is_available():
            log.warning("gpu is not available, use cpu")
            args.device = "cpu"

    if args.device not in DEVICE_LIST:
        log.error(
            f"unkown device: {args.device}, only thess is available: {DEVICE_LIST}"
        )
        exit(-1)

    if args.segment_duration < MIN_SEGMENT_LENGTH:
        log.error(f"segment duration can not smaller than {MIN_SEGMENT_LENGTH}")
        exit(-1)


def init_log():
    log.remove()
    log.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
    )


def _generate_imgs(
    ffmpeg_bin_file_path: str, videos_dir_path: str, output_dir: str, fps: int
):
    videos_files_path = [
        os.path.join(videos_dir_path, img_name)
        for img_name in os.listdir(videos_dir_path)
    ]

    time_recorder = TimeRecorder()
    time_recorder.start_record()
    for video_file_path in videos_files_path:
        log.info(f"start generate imgs of {video_file_path}")
        generate_imgs(ffmpeg_bin_file_path, video_file_path, output_dir, fps)
    time_recorder.end_record()
    log.info(
        f"generate imgs of {len(videos_files_path)} videos cost {time_recorder.get_duration_miliseconds()} ms"
    )


def _save_feats(feats_save_path, feats):
    tmp_dir = os.path.dirname(feats_save_path)
    os.makedirs(tmp_dir, exist_ok=True)
    np.save(feats_save_path, feats)


def _generate_feats(imgs_output_dir: str, isc_weight_path: str, device: str):
    isc_model, isc_processer = create_isc_model(
        weight_file_path=isc_weight_path, device=device, is_training=False
    )

    feats_time_recorder = TimeRecorder()
    save_time_recorder = TimeRecorder()
    imgs_count = 0

    tmp_imgs_output_dir = [
        os.path.join(imgs_output_dir, title) for title in os.listdir(imgs_output_dir)
    ]

    for img_dir_path in tmp_imgs_output_dir:
        rimgs_list = [
            os.path.join(img_dir_path, img_name)
            for img_name in os.listdir(img_dir_path)
        ]

        imgs_count += len(rimgs_list)

        log.info(f"start gen feats, imgs len: {len(rimgs_list)}")
        # it's very slow when use cpu to generate image feats....
        feats_time_recorder.start_record()
        ref_isc_feats = gen_img_feats_by_ISCNet(
            rimgs_list, isc_model, isc_processer, device
        )
        feats_time_recorder.end_record()
        log.info(f"get feats: {ref_isc_feats.shape}")

        ref_feats_file_path = os.path.join(
            ref_feats_output_dir,
            f"{os.path.splitext(os.path.basename(img_dir_path))[0]}.npy",
        )

        save_time_recorder.start_record()
        _save_feats(ref_feats_file_path, ref_isc_feats)
        save_time_recorder.end_record()

        log.info(f"save feats to {ref_feats_file_path} ")

    log.info(
        f"generate iscNet feats by {device} of {imgs_count} imgs cost {feats_time_recorder.get_duration_miliseconds()} ms, save it cost {save_time_recorder.get_duration_miliseconds()} ms"
    )


if __name__ == "__main__":
    # torch.set_num_threads(16)
    init_log()
    args = parser_args()
    check_args(args)

    ref_videos_dir_path = args.reference_videos_dir
    sample_videos_dir_path = args.sample_videos_dir

    device = args.device
    ffmpeg_path = args.ffmpeg
    output_dir = args.output_dir

    isc_weight_path = args.isc_weight
    transvcl_weight_path = args.transVCL_weight

    confthre = args.conf_thre
    nmsthre = args.nms_thre
    img_size = (args.img_size, args.img_size)

    segment_duration: int = args.segment_duration

    fps = args.fps
    frame_interval = 1000.0 / float(fps)

    if frame_interval > segment_duration:
        log.warning(
            f"fps is {fps}, frame interval {frame_interval} ms is bigger than segment duration: {segment_duration} ms"
        )
        segment_duration = round(frame_interval * 10)
        log.warning(f"segment duration reset to {segment_duration} ms")

    segment_length = round(segment_duration / frame_interval)
    log.info(
        f"segment_length: {segment_length}, segment_duration: {segment_duration}, frame_interval: {frame_interval}"
    )

    # 1. 解析输入媒体文件列表，判断是否重新提取帧
    re_extract_imgs_flag = True

    ref_output_base_name = f"{os.path.basename(ref_videos_dir_path)}_fps_{fps}"
    sample_output_base_name = f"{os.path.basename(sample_videos_dir_path)}_fps_{fps}"

    ref_imgs_output_dir = os.path.join(output_dir, "imgs", ref_output_base_name)
    sample_imgs_output_dir = os.path.join(output_dir, "imgs", sample_output_base_name)

    if re_extract_imgs_flag:
        log.info("star generate ref imgs...")
        _generate_imgs(ffmpeg_path, ref_videos_dir_path, ref_imgs_output_dir, fps)

        log.info("star generate sample imgs...")
        _generate_imgs(ffmpeg_path, sample_videos_dir_path, sample_imgs_output_dir, fps)

    else:
        log.info(f"use cached imgs: {ref_imgs_output_dir} and {sample_imgs_output_dir}")

    # 2. 使用 IscNet 提取特征
    re_extract_imgs_feats_flag = True

    ref_feats_output_dir = os.path.join(output_dir, "feats", ref_output_base_name)
    sample_feats_output_dir = os.path.join(output_dir, "feats", sample_output_base_name)

    if re_extract_imgs_feats_flag:
        log.info("star generate ref feats...")
        _generate_feats(ref_imgs_output_dir, isc_weight_path, device)

        log.info("star generate sample feats...")
        _generate_feats(sample_imgs_output_dir, isc_weight_path, device)

    else:
        log.info(
            f"use cached feats: {ref_feats_output_dir} and {sample_feats_output_dir}"
        )
