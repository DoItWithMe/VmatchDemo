from exception import VmatchException
from functools import wraps


class ModelException(VmatchException):
    """milvus exception"""

    pass


def _exception_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OSError as e:
            raise ModelException(f"models have os exception: {e}")
        except ValueError as e:
            raise ModelException(f"models have input value error: {e}")
        except RuntimeError as e:
            raise ModelException(f"models have input value error: {e}")
        except ModelException as e:
            raise e
        except Exception as e:
            raise ModelException(f"models have exception: {e}")

    return wrapper
