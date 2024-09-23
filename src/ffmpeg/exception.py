from exception import VmatchException
from functools import wraps


class FFmpegException(VmatchException):
    """milvus exception"""

    pass


def _exception_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OSError as e:
            raise FFmpegException(f"ffmpeg have os exception: {e}")
        except ValueError as e:
            raise FFmpegException(f"ffmpeg have input value error: {e}")
        except FFmpegException as e:
            raise e
        except Exception as e:
            raise FFmpegException(f"ffmpeg have exception: {e}")

    return wrapper
