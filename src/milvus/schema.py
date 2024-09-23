from pymilvus import FieldSchema, DataType, CollectionSchema
from pymilvus.milvus_client import IndexParams

EMBEDDING_COLLECTION_NAME = "vmatch_video_embeddings"

EMBEDDING_FIELDS = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="frame", dtype=DataType.INT64, is_primary=False),
    FieldSchema(
        name="uuid",
        dtype=DataType.VARCHAR,
        max_length=1024,
    ),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=256),
]

EMBEDDING_SCHEMA = CollectionSchema(
    fields=EMBEDDING_FIELDS,
    description="vmatch video embeddings schema",
    auto_id=True,
)

# TODO:
# milvus 不会改变 nlist 的数量，而这个又影响向量数据库的搜索，所以需要考虑如何自动化维护 nlist
# 封装下 milvus 的各项操作

EMBEDDING_INDEX_FIELD_NAME = "embedding"
EMBEDDING_INDEX_TYPE = "IVF_SQ8"
EMBEDDING_INDEX_NAME = "video_embedding_ivf_sq8"
EMBEDDING_INDEX_METRIC_TYPE = "L2"

EMBEDDING_INDEX_NLIST = 128

EMBEDDING_INDEX_PARAMS = IndexParams()
EMBEDDING_INDEX_PARAMS.add_index(
    field_name=EMBEDDING_INDEX_FIELD_NAME,
    index_type=EMBEDDING_INDEX_TYPE,
    index_name=EMBEDDING_INDEX_NAME,
    metric_type=EMBEDDING_INDEX_METRIC_TYPE,
    nlist=EMBEDDING_INDEX_NLIST,
)

MEDIA_INFO_COLLECTION_NAME = "media_info"
MEDIA_INFO_FIELDS = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="frame_total_len", dtype=DataType.INT64),
    FieldSchema(
        name="uuid",
        dtype=DataType.VARCHAR,
        max_length=1024,
    ),
]

MEDIA_INFO_SCHEMA = CollectionSchema(
    fields=MEDIA_INFO_FIELDS,
    description="vmatch video media info schema",
    auto_id=True,
)
