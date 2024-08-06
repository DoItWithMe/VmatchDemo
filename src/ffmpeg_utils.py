import subprocess
import os

# import shutil
import signal


def run_ffmpeg(
    ffmpeg_bin_file_path: str, input_file_path: str, output_imgs_dir_path: str
):
    """
    :param input_file_path: 输入视频文件路径
    :param output_imgs_dir_path: 输出视频帧存储路径
    :return: ValueError or None
    """

    if not os.path.exists(ffmpeg_bin_file_path):
        return ValueError("ffmpeg not exists")

    if not os.path.exists(input_file_path):
        return ValueError("input file not exists")
    try:
        output_imgs_dir_path = os.path.join(
            output_imgs_dir_path, os.path.splitext(os.path.basename(input_file_path))[0]
        )
        os.makedirs(output_imgs_dir_path, exist_ok=True)
    except Exception as e:
        return ValueError(f"create output dir: {output_imgs_dir_path} failed for {e}")

    command = [
        ffmpeg_bin_file_path,
        "-i",
        input_file_path,
        "-nostdin -y -vf fps=1 -start_number 0 -q 0",
        f"{output_imgs_dir_path}/%05d.jpg",
    ]

    try:
        ffmpeg_p = subprocess.Popen(
            command,
            bufsize=0,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            close_fds=True,
            preexec_fn=os.setsid,
        )

        ffmpeg_log_list: list[str] = list()
        for line in iter(ffmpeg_p.stdout.readline, b""):  # type: ignore
            ffmpeg_log_list.append(line)

        ffmpeg_p.stdout.close()  # type: ignore
        try:
            os.killpg(os.getpgid(ffmpeg_p.pid), signal.SIGTERM)
        except OSError as error:
            pass

    except Exception as e:
        return ValueError(f"run ffmpeg cmd: {command} failed for {e}")
    return None
