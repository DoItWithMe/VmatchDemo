from exception import VmatchException
from functools import wraps
from pymilvus import exceptions as pymilvus_exceptions

class MilvusException(VmatchException):
    """milvus exception"""

    pass


def _exception_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except pymilvus_exceptions.MilvusException as e:
            raise MilvusException(f"milvus have exception: {e.message}")
        except MilvusException as e:
            raise e
        except Exception as e:
            raise MilvusException(f"milvus have exception: {e}")

    return wrapper
