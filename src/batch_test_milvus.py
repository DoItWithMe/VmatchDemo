import sys
import os

_project_dipath: str = os.path.dirname((os.path.abspath(__file__)))
sys.path.append(_project_dipath)

from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema
from pymilvus.milvus_client import IndexParams

from loguru import logger as log
import argparse
from typing import Any
from utils.time_utils import TimeRecorder
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import uuid

from results_handler.query_results_handler import handler_query_results


def init_log():
    log.remove()
    log.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
    )


def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        help="output dir path, auto-create it if not exisit",
        default="./output",
        required=False,
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


def create_milvus_client(
    embedding_collection_name: str, media_info_collection_name: str
):
    index_type = "FLAT"
    nlist = 4096
    metric_type = "L2"

    local_db_name = f"milvus_match_demo_{index_type}_{nlist}_{metric_type}.db"
    milvus_client = MilvusClient(local_db_name)
    return milvus_client

    if milvus_client.has_collection(collection_name=embedding_collection_name):
        milvus_client.drop_collection(collection_name=embedding_collection_name)

    embedding_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="frame", dtype=DataType.INT64, is_primary=False),
        FieldSchema(
            name="uuid",
            dtype=DataType.VARCHAR,
            max_length=1024,
        ),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=256),
    ]

    embedding_schema = CollectionSchema(
        embedding_fields, "video embeddings", auto_id=True
    )

    embedding_index_params = IndexParams()
    embedding_index_params.add_index(
        field_name="embedding",
        index_type=index_type,
        index_name="video_embedding_ivf_sq8",
        metric_type=metric_type,
        nlist=nlist,
    )

    milvus_client.create_collection(
        collection_name=embedding_collection_name,
        schema=embedding_schema,
        index_params=embedding_index_params,
    )

    if milvus_client.has_collection(collection_name=media_info_collection_name):
        milvus_client.drop_collection(collection_name=media_info_collection_name)

    media_info_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="frame_total_len", dtype=DataType.INT64),
        FieldSchema(
            name="uuid",
            dtype=DataType.VARCHAR,
            max_length=1024,
        ),
    ]
    media_info_schema = CollectionSchema(
        media_info_fields, "video media info", auto_id=True
    )

    milvus_client.create_collection(
        collection_name=media_info_collection_name, schema=media_info_schema
    )

    return milvus_client


def _load_feats(feats_store_dir_path: str):
    time_recorder = TimeRecorder()
    feat_dict: dict[str, Any] = dict()
    feats_files_list = [
        os.path.join(feats_store_dir_path, file_name)
        for file_name in os.listdir(feats_store_dir_path)
    ]

    time_recorder.start_record()
    count = 0
    for feats_file in feats_files_list:
        feat_dict[os.path.splitext(os.path.basename(feats_file))[0]] = np.load(
            feats_file
        )
        count += 1
    time_recorder.end_record()
    log.info(
        f"load feats of {feats_store_dir_path} cost {time_recorder.get_total_duration_miliseconds()} ms"
    )

    return feat_dict


def _add_ref(
    milvus_client: MilvusClient,
    embedding_collection_name: str,
    media_info_db_name: str,
    ref_feat_dict: dict[str, Any],
):
    for ref_name, ref_feats in ref_feat_dict.items():
        ref_uuid = str(uuid.uuid4())

        log.info(f"start add ref embeddings of {ref_name}")
        milvus_client.insert(
            media_info_db_name,
            {
                "uuid": ref_uuid,
                "filename": ref_name,
                "frame_total_len": len(ref_feats),
            },
        )

        for i, feat in enumerate(ref_feats):
            if i % 1000 == 0:
                log.info(f"insert {i}/{len(ref_feats)}..")
            milvus_client.insert(
                embedding_collection_name,
                {
                    "frame": i,
                    "uuid": ref_uuid,
                    "embedding": feat,
                },
            )


def _denormalize(tensor, mean, std):
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)

    # Create a new tensor for denormalized values
    denormalized_tensor = tensor.clone()

    # Perform denormalization
    for t, m, s in zip(denormalized_tensor, mean, std):
        t.mul_(s).add_(m)

    return denormalized_tensor


def _draw_plot(
    ref_output_base_name: str,
    sample_output_base_name: str,
    formatted_sample_number: str,
    sample_name: str,
    results: list[list[dict[Any, Any]]],
    topK: int,
):
    base_width = 512
    base_height = 512
    width = base_width * 5
    height = base_height * 3

    sample_img = Image.open(
        os.path.join(
            "/data/jinzijian/VmatchDemo",
            "output",
            "imgs",
            sample_output_base_name,
            sample_name,
            f"{formatted_sample_number}.jpg",
        )
    ).resize((base_width, base_height))

    for result in results:
        log.info(f"res: {result}")
        images = []
        for hit in result[:topK]:
            formatted_hit_number = str(hit["id"]).zfill(6)
            hit_img = Image.open(
                os.path.join(
                    "/data/jinzijian/VmatchDemo",
                    "output",
                    "imgs",
                    ref_output_base_name,
                    hit["entity"]["filename"],
                    f"{formatted_hit_number}.jpg",
                )
            ).resize((base_width, base_height))
            images.append(hit_img)

        concatenated_image = Image.new("RGB", (width, height))
        concatenated_image.paste(sample_img, (0, 0))
        for idx, img in enumerate(images):
            x = idx % 5
            y = idx // 5 + 1
            concatenated_image.paste(img, (x * base_width, y * base_height))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(concatenated_image)
        ax.axis("off")
        os.makedirs(
            os.path.join("/data/jinzijian/VmatchDemo", "tmp", sample_name),
            exist_ok=True,
        )
        plt.savefig(
            os.path.join(
                "/data/jinzijian/VmatchDemo",
                "tmp",
                sample_name,
                f"{formatted_sample_number}.png",
            ),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()


def _query(
    milvus_client: MilvusClient,
    collection_name: str,
    media_info_collection_name: str,
    sample_feat_dict: dict[str, Any],
    topK: int,
    ref_output_base_name: str,
    sample_output_base_name: str,
    l2_dis_thresh: float,
):

    segment_len_limit = 8 * 3
    sample_name_list = list(sample_feat_dict.keys())
    sample_name_list = sorted(sample_name_list, key=lambda x: int(x.split("_")[1]))
    for sample_name in sample_name_list:
        sample_feat = sample_feat_dict[sample_name]
        # if sample_name != "R_2_jieshuo" and sample_name != "R_2_jieshuo":
        #     continue

        results = milvus_client.search(
            collection_name,
            data=sample_feat,
            output_fields=["frame", "uuid"],
            search_params={"metric_type": "L2"},
            limit=3 * segment_len_limit,
        )

        handler_query_results(
            milvus_client,
            media_info_collection_name,
            l2_dis_thresh,
            results,
            segment_len_limit,
            sample_name,
        )
        # break


def main():
    init_log()

    # create milvus client
    embedding_collection_name = "video_embeddings"
    media_info_collection_name = "media_info"

    milvus_client = create_milvus_client(
        embedding_collection_name, media_info_collection_name
    )
    # 再添加一个 collection 用于存储 media_info

    #
    args = parser_args()
    fps = 8
    output_dir = args.output_dir

    ref_videos_dir_path = os.path.normpath(args.reference_videos_dir)
    sample_videos_dir_path = os.path.normpath(args.sample_videos_dir)

    ref_output_base_name = f"{os.path.basename(ref_videos_dir_path)}_fps_{fps}"
    sample_output_base_name = f"{os.path.basename(sample_videos_dir_path)}_fps_{fps}"

    ref_feats_output_dir = os.path.join(output_dir, "feats", ref_output_base_name)
    sample_feats_output_dir = os.path.join(output_dir, "feats", sample_output_base_name)

    ref_feat_dict = _load_feats(ref_feats_output_dir)

    # _add_ref(
    #     milvus_client,
    #     embedding_collection_name,
    #     media_info_collection_name,
    #     ref_feat_dict,
    # )

    sample_feat_dict = _load_feats(sample_feats_output_dir)

    l2_dis_thresh = 1.1
    _query(
        milvus_client,
        embedding_collection_name,
        media_info_collection_name,
        sample_feat_dict,
        2 * fps,
        ref_output_base_name,
        sample_output_base_name,
        l2_dis_thresh,
    )


if __name__ == "__main__":
    main()
