# 720p to 1080p Video Upscaler

This project provides a Python script to upscale videos using the Real-ESRGAN model.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install system dependencies:**
    This script requires `ffmpeg` and `ffprobe` to be installed on your system. You can install them using your system's package manager.

    *   **On Debian/Ubuntu:**
        ```bash
        sudo apt-get update
        sudo apt-get install ffmpeg
        ```

    *   **On macOS (using Homebrew):**
        ```bash
        brew install ffmpeg
        ```

    *   **On Windows:**
        Download the binaries from the official FFmpeg website and add them to your system's PATH.

4.  **Download Real-ESRGAN Models:**
    You need to download the pre-trained Real-ESRGAN models and place them in the `~/.cache/realesrgan/` directory.

    *   **For 4x upscaling:**
        ```bash
        wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ~/.cache/realesrgan/
        ```

    *   **For 2x upscaling:**
        ```bash
        wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -P ~/.cache/realesrgan/
        ```

## Usage

To upscale a video, run the `upscale_video.py` script with the input and output file paths as arguments. You can also specify the upscaling factor using the `--scale` argument.

```bash
python upscale_video.py <input_video> <output_video> [--scale <2_or_4>]
```

For example, to upscale a video using the 2x model:

```bash
python upscale_video.py my_720p_video.mp4 my_1080p_video.mp4 --scale 2
```

If you don't provide the `--scale` argument, it will default to 4x.

The script will then:
1.  Extract the frames and audio from the input video.
2.  Upscale each frame using the selected Real-ESRGAN model.
3.  Reassemble the upscaled frames with the original audio into a new video file.
