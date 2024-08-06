
import argparse
from ffmpeg_utils import run_ffmpeg

def parser_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ffmpeg", type=str, help="ffmpeg bin file path", default="./assets/ffmpeg")
    parser.add_argument("--isc-model", type=str, help="isc model path", default="./assets/models/isc_ft_v107.pth.tar")
    parser.add_argument("--transVCS-model", type=str, help="transVCS model path", default="./assets/models/tarnsVCL_model_1.pth")
    
    parser.add_argument("--output-dir", type=str, help="output dir path, auto-create it if not exisit", default="./output")
    
    parser.add_argument("--input-sample-file-path", "-s", type=str, help="sample media file path")
    parser.add_argument("--input-reference-file-path", "-r", type=str, help="reference media file path")
    
    return parser.parse_args()

if __name__ == '__main__':
    print("hi")
    args = parser_args()
    
