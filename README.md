# High-Quality Video and Image Super-Resolution

This project provides a powerful tool for upscaling videos and images to higher resolutions using advanced AI models. It leverages Real-ESRGAN to achieve state-of-the-art results in super-resolution, enhancing the quality and detail of your media.

## Features

- **High-Quality Upscaling:** Utilizes Real-ESRGAN, a cutting-edge super-resolution model, to produce sharp and detailed results.
- **Video and Image Support:** Upscale both videos (MP4, MKV, AVI, MOV) and images (PNG, JPG, WEBP).
- **Flexible Resolution Control:** Choose between a specific scaling factor (e.g., 2x, 4x) or a target resolution (e.g., 1080p, 4k).
- **Hardware Accelerated:** Leverages NVIDIA GPUs for fast processing.
- **Audio Preservation:** Automatically preserves the original audio track in upscaled videos.

## Prerequisites

- **Python 3.8+**
- **NVIDIA GPU** with CUDA and cuDNN installed.
- **ffmpeg:** Required for video processing. You can install it using your system's package manager (e.g., `sudo apt-get install ffmpeg` on Debian/Ubuntu).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install the required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the pre-trained models:**
   Download the `RealESRGAN_x4plus.pth` and `RealESRGAN_x2plus.pth` models and place them in the `models` directory. You can find the models in the original Real-ESRGAN repository or other trusted sources.

## Usage

### Upscaling a Single File

To upscale a single image or video, use the `super_resolution.py` script with the `input_path` and `output_path` arguments.

**Example (Upscale to 4k):**

```bash
python super_resolution.py path/to/your/input.mp4 path/to/your/output.mp4 --target-resolution 4k
```

**Example (Upscale by a factor of 2):**

```bash
python super_resolution.py path/to/your/image.jpg path/to/your/output.jpg --scale 2
```

### Upscaling a Directory

You can also process all files in a directory by providing the input and output directory paths.

```bash
python super_resolution.py path/to/your/input_directory path/to/your/output_directory --target-resolution 1080p
```

### Command-Line Arguments

- `input_path`: Path to the input video, image, or directory.
- `output_path`: Path to the output file or directory.
- `--scale`: The upscaling factor (e.g., 1.5, 2.0). Mutually exclusive with `--target-resolution`.
- `--target-resolution`: The target resolution (e.g., 1080p, 2k, 4k, 5k). Mutually exclusive with `--scale`.

## Model Conversion

The `convert_x4_model.py` script can be used to convert a x4 Real-ESRGAN model to x2 and x1 models.

```bash
python convert_x4_model.py
```

This will create `RealESRGAN_x2plus.pth` and `RealESRGAN_x1plus.pth` from `RealESRGAN_x4plus.pth` in the `models` directory.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is built upon the excellent work of the following open-source projects:

- **Real-ESRGAN:** [https://github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **BasicSR:** [https://github.com/xinntao/BasicSR](https://github.com/xinntao/BasicSR)