import subprocess
import os

# import shutil
import signal


def generate_1fps_imgs(
    ffmpeg_bin_file_path: str, input_file_path: str, output_imgs_dir_path: str
) -> str:
    """
    Generate 1fps images from media by FFmpeg
        
    Args:
        ffmpeg_bin_file_path (`str`):
            ffmpeg bin path
        input_file_path (`str`):
            input media file path
        output_imgs_dir_path (`str`):
            output images local store dir
    
    Returns:
        tmp_output_imgs_dir_path (`str`):
            output images local store dir named as "output_imgs_dir_path/input_file_name"
    """

    if not os.path.exists(ffmpeg_bin_file_path):
        raise ValueError("ffmpeg not exists")

    if not os.path.exists(input_file_path):
        raise ValueError("input file not exists")
    try:
        tmp_output_imgs_dir_path:str = os.path.join(
            output_imgs_dir_path, f"{os.path.splitext(os.path.basename(input_file_path))[0]}_imgs"
        )
        os.makedirs(tmp_output_imgs_dir_path, exist_ok=True)
    except Exception as e:
        raise ValueError(f"create output dir: {tmp_output_imgs_dir_path} failed for {e}")

    command = f"{ffmpeg_bin_file_path} -i {input_file_path} -loglevel error -nostdin -y -vf fps=1 -start_number 0 -q 0 {tmp_output_imgs_dir_path}/%05d.jpg"

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

        # ffmpeg_log_list: list[str] = list()
        # for line in iter(ffmpeg_p.stdout.readline, b""):  # type: ignore
        #     ffmpeg_log_list.append(line)
            # print(line)
            
        stdout, _ = ffmpeg_p.communicate()
        if 0 != ffmpeg_p.returncode:
            raise RuntimeError(f"ffmpeg run failed for {stdout}")

        try:
            os.killpg(os.getpgid(ffmpeg_p.pid), signal.SIGTERM)
        except OSError as error:
            pass

    except Exception as e:
        raise ValueError(f"run ffmpeg cmd: {command} failed for {e}")
    
    return tmp_output_imgs_dir_path
