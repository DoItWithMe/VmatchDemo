from pymilvus import MilvusClient, CollectionSchema
from pymilvus.milvus_client import IndexParams
from pydantic import BaseModel
from typing import Optional, Any, Union, List
from numpy.typing import NDArray
from .schema import *
import uuid
from .exception import MilvusException, _exception_handler


class MilvusClientConfig(BaseModel):
    uri: str = ""
    usr: str = ""
    password: str = ""
    db_name: str = ""
    token: str = ""

    # timeout unit is seconds
    timeout: Optional[float] = 10

    def __str__(self):
        return f"uri: {self.uri}, usr: {self.usr}, password: {self.password}, db_name: {self.db_name}, timeout: {self.timeout} s"


class MilvusClientManager:
    @_exception_handler
    def __init__(self, milvus_client_cfg: MilvusClientConfig) -> None:
        # 异常捕获
        self._milvus_client_cfg = milvus_client_cfg
        self._milvus_client = MilvusClient(
            uri=self._milvus_client_cfg.uri,
            usr=self._milvus_client_cfg.usr,
            password=self._milvus_client_cfg.password,
            db_name=self._milvus_client_cfg.db_name,
            token=self._milvus_client_cfg.token,
            timeout=self._milvus_client_cfg.timeout,
        )

        self._check_collection(
            EMBEDDING_COLLECTION_NAME,
            EMBEDDING_SCHEMA,
            EMBEDDING_INDEX_NAME,
            EMBEDDING_INDEX_PARAMS,
        )

        self._check_collection(
            MEDIA_INFO_COLLECTION_NAME,
            MEDIA_INFO_SCHEMA,
            None,
            None,
        )

    @_exception_handler
    def _check_collection(
        self,
        collection_name: str,
        schema: CollectionSchema,
        index_name: Optional[str],
        index_params: Optional[IndexParams],
    ):
        if not self._milvus_client.has_collection(collection_name=collection_name):
            raise MilvusException(f"not have collection: {collection_name}")

        if index_name is not None and index_params is not None:
            index_info = self._milvus_client.describe_index(
                collection_name=collection_name, index_name=index_name
            )
            if not index_info:
                raise MilvusException(
                    f"collection: {collection_name} not have index: {index_name} with index_params: {index_params}"
                )

    @_exception_handler
    def add_ref_embedding(self, ref_name: str, ref_embeddings: NDArray[Any]):
        ref_uuid = str(uuid.uuid4())
        self._milvus_client.insert(
            collection_name=MEDIA_INFO_COLLECTION_NAME,
            data={
                "uuid": ref_uuid,
                "filename": ref_name,
                "frame_total_len": len(ref_embeddings),
            },
        )

        for i, feat in enumerate(ref_embeddings):
            self._milvus_client.insert(
                collection_name=EMBEDDING_COLLECTION_NAME,
                data={
                    "frame": i,
                    "uuid": ref_uuid,
                    "embedding": feat,
                },
            )

    @_exception_handler
    def search_matched_embeddings(
        self, sample_embeddings: Union[List[list], list], limit: int = 10
    ):
        results = self._milvus_client.search(
            collection_name=EMBEDDING_COLLECTION_NAME,
            data=sample_embeddings,
            output_fields=["frame", "uuid"],
            search_params={"metric_type": EMBEDDING_INDEX_METRIC_TYPE},
            limit=limit,
        )

        return results


# TODO: 配置文件
milvus_client_cfg = MilvusClientConfig(uri="./milvus_match_demo_2.db")
milvus_client_manager = MilvusClientManager(milvus_client_cfg)
