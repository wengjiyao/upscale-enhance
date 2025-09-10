

import os
import subprocess
import tempfile
import argparse
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

RESOLUTION_MAP = {
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
    "5k": (5120, 2880),
}

def get_input_resolution(input_path):
    """Gets the resolution of an image or video file."""
    input_ext = os.path.splitext(input_path)[1].lower()
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov']
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']

    if input_ext in image_extensions:
        img = cv2.imread(input_path)
        if img is not None:
            return img.shape[1], img.shape[0]
    elif input_ext in video_extensions:
        try:
            command = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0",
                input_path,
            ]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            width, height = map(int, result.stdout.strip().split('x'))
            return width, height
        except Exception as e:
            print(f"Error getting video resolution: {e}")
    return None

def get_video_framerate(video_path):
    """Gets the frame rate of a video using ffprobe."""
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        num, den = map(int, result.stdout.split('/'))
        return num / den
    except Exception as e:
        print(f"Error getting frame rate: {e}")
        raise


def has_audio_stream(video_path):
    """Checks if a video file has an audio stream."""
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        return result.stdout.strip() != ""
    except subprocess.CalledProcessError as e:
        print(f"Error checking for audio stream: {e.stderr}")
        return False


def initialize_upsampler(scale):
    """Initializes the Real-ESRGAN upsampler."""
    if scale == 4:
        model_path = os.path.join(os.path.expanduser("~"), ".cache/realesrgan/RealESRGAN_x4plus.pth")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    elif scale == 2:
        model_path = os.path.join(os.path.expanduser("~"), ".cache/realesrgan/RealESRGAN_x2plus.pth")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    else:
        print("Error: Invalid scale factor. Please choose 2 or 4.")
        return None

    print(f"Initializing Real-ESRGAN model for {scale}x upscaling...")
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=0
    )
    return upsampler


def upscale_image(input_path, output_path, upsampler, target_resolution):
    """Upscales a single image and resizes to target resolution."""
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Cannot read image at {input_path}")
        return

    try:
        upscaled_img, _ = upsampler.enhance(img)
        final_img = cv2.resize(upscaled_img, target_resolution, interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(output_path, final_img)
        print(f"Successfully upscaled image to {output_path}")
    except Exception as e:
        print(f"Error upscaling image {input_path}: {e}")


def upscale_video(input_path, output_path, upsampler, target_resolution):
    """
    Upscales a video using Real-ESRGAN.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return


    with tempfile.TemporaryDirectory() as temp_dir:
        frames_dir = os.path.join(temp_dir, "frames")
        upscaled_frames_dir = os.path.join(temp_dir, "upscaled_frames")
        os.makedirs(frames_dir)
        os.makedirs(upscaled_frames_dir)

        audio_path = os.path.join(temp_dir, "audio.aac")
        has_audio = has_audio_stream(input_path)
        
        print(f"Extracting frames from {input_path}...")
        try:
            # Extract frames
            subprocess.run(
                ["ffmpeg", "-i", input_path, "-q:v", "1", "-pix_fmt", "rgb24", f"{frames_dir}/frame_%06d.png"],
                check=True, capture_output=True, text=True
            )
            if has_audio:
                print("Extracting audio...")
                # Extract audio
                subprocess.run(
                    ["ffmpeg", "-i", input_path, "-vn", "-acodec", "copy", audio_path],
                    check=True, capture_output=True, text=True
                )
        except subprocess.CalledProcessError as e:
            print("Error during ffmpeg extraction:")
            print(e.stderr)
            return

        frame_files = sorted(os.listdir(frames_dir))
        total_frames = len(frame_files)
        print(f"Successfully extracted {total_frames} frames.")

        print("Starting frame upscaling...")
        for i, frame_name in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, frame_name)
            img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            
            try:
                upscaled_img, _ = upsampler.enhance(img)
                final_img = cv2.resize(upscaled_img, target_resolution, interpolation=cv2.INTER_LANCZOS4)
                save_path = os.path.join(upscaled_frames_dir, frame_name)
                cv2.imwrite(save_path, final_img)
            except Exception as e:
                print(f"Error upscaling frame {frame_name}: {e}")
                return # Stop processing
            
            print(f"Upscaled frame {i + 1}/{total_frames}", end="\r")

        print("\nUpscaling complete.")
        
        print("Reassembling video with upscaled frames and original audio...")
        try:
            framerate = get_video_framerate(input_path)
            reassemble_command = [
                "ffmpeg",
                "-r", str(framerate),
                "-i", f"{upscaled_frames_dir}/frame_%06d.png",
            ]
            if has_audio:
                reassemble_command.extend(["-i", audio_path, "-c:a", "copy"])
            
            reassemble_command.extend([
                "-c:v", "libx264",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                output_path,
            ])
            subprocess.run(reassemble_command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("Error during ffmpeg reassembly:")
            print(e.stderr)
            return

    print(f"Successfully created upscaled video at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale a video or image using Real-ESRGAN.")
    parser.add_argument("input_path", help="Path to the input video or image file.")
    parser.add_argument("output_path", help="Path to save the output file.")
    parser.add_argument("--scale", type=float, help="Upscaling factor (e.g., 1.5, 2.0). Mutually exclusive with --target-resolution.")
    parser.add_argument("--target-resolution", type=str, help="Target resolution (e.g., 1080p, 4k). Mutually exclusive with --scale.")
    
    args = parser.parse_args()

    if args.scale and args.target_resolution:
        print("Error: Please provide either --scale or --target-resolution, not both.")
        exit()

    if not args.scale and not args.target_resolution:
        print("Error: Please provide either --scale or --target-resolution.")
        exit()

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    input_resolution = get_input_resolution(args.input_path)
    if input_resolution is None:
        print("Error: Could not determine input resolution.")
        exit()

    if args.target_resolution:
        if args.target_resolution in RESOLUTION_MAP:
            target_resolution = RESOLUTION_MAP[args.target_resolution]
            scale_factor = target_resolution[0] / input_resolution[0]
        else:
            print("Error: Invalid target resolution.")
            exit()
    else:
        scale_factor = args.scale
        target_resolution = (int(input_resolution[0] * scale_factor), int(input_resolution[1] * scale_factor))

    if scale_factor <= 2:
        model_scale = 2
    elif scale_factor <= 4:
        model_scale = 4
    else:
        print("Error: Scaling factors greater than 4 are not yet supported.")
        exit()

    upsampler = initialize_upsampler(model_scale)
    if upsampler is None:
        exit()

    # Determine if the input is a video or an image based on file extension
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov']
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    
    input_ext = os.path.splitext(args.input_path)[1].lower()

    if input_ext in video_extensions:
        upscale_video(args.input_path, args.output_path, upsampler, target_resolution)
    elif input_ext in image_extensions:
        upscale_image(args.input_path, args.output_path, upsampler, target_resolution)
    else:
        print("Error: Unsupported file format. Please provide a video or image file.")

