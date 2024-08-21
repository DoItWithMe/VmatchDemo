# VmatchDemo

基于以下开源仓库调研如何开发视频比对小工具
[FFmpeg](https://github.com/FFmpeg/FFmpeg)
[ISC 图像特征提取](https://github.com/lyakaap/ISC21-Descriptor-Track-1st)
[VSCL 数据集](https://github.com/ant-research/VCSL?tab=readme-ov-file)
[TransVCL 侵权定位算法](https://github.com/transvcl/TransVCL)


```流式读取 FFmpeg 输出,PNG
import subprocess

def split_pngs(data):
    start_marker = b'\x89PNG\r\n\x1a\n'
    end_marker = b'IEND\xaeB`\x82'
    images = []
    start = 0
    
    while True:
        start = data.find(start_marker, start)
        if start == -1:
            break
        end = data.find(end_marker, start) + len(end_marker)
        images.append(data[start:end])
        start = end

    return images

# 启动 FFmpeg 进程
ffmpeg_process = subprocess.Popen(
    ['ffmpeg', '-i', 'input.mp4', '-vf', 'fps=1', '-f', 'image2pipe', '-vcodec', 'png', '-'],
    stdout=subprocess.PIPE
)

# 读取并分割图片
buffer = b''
while True:
    chunk = ffmpeg_process.stdout.read(1024)
    if not chunk:
        break
    buffer += chunk

    images = split_pngs(buffer)
    for i, img_data in enumerate(images):
        with open(f'image_{i}.png', 'wb') as img_file:
            img_file.write(img_data)
    
    # 删除已处理的部分
    last_image_end = buffer.rfind(b'IEND\xaeB`\x82') + len(b'IEND\xaeB`\x82')
    buffer = buffer[last_image_end:]

```


```流式读取 FFmpeg 输出，jpeg
import subprocess

def split_jpegs(data):
    start_marker = b'\xFF\xD8'
    end_marker = b'\xFF\xD9'
    images = []
    start = 0
    
    while True:
        start = data.find(start_marker, start)
        if start == -1:
            break
        end = data.find(end_marker, start) + len(end_marker)
        if end == -1:
            break
        images.append(data[start:end])
        start = end

    return images

# 启动 FFmpeg 进程
ffmpeg_process = subprocess.Popen(
    ['ffmpeg', '-i', 'input.mp4', '-vf', 'fps=1', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-'],
    stdout=subprocess.PIPE
)

# 读取并分割图片
buffer = b''
while True:
    chunk = ffmpeg_process.stdout.read(1024)
    if not chunk:
        break
    buffer += chunk

    images = split_jpegs(buffer)
    for i, img_data in enumerate(images):
        with open(f'image_{i}.jpg', 'wb') as img_file:
            img_file.write(img_data)
    
    # 删除已处理的部分
    last_image_end = buffer.rfind(b'\xFF\xD9') + len(b'\xFF\xD9')
    buffer = buffer[last_image_end:]

```