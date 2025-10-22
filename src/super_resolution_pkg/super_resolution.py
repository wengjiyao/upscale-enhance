import os
import subprocess
import tempfile
import argparse
import cv2
import shutil
import time
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

MIN_REQUIRED_DISK_SPACE_GB = 50 # Minimum required disk space in GB

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
    base_model_dir = os.path.join(os.path.dirname(__file__), "models")
    if scale == 4:
        model_path = os.path.join(base_model_dir, "RealESRGAN_x4plus.pth")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    elif scale == 2:
        model_path = os.path.join(base_model_dir, "RealESRGAN_x2plus.pth")
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
    start_time = time.time()
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Cannot read image at {input_path}")
        return

    try:
        upscaled_img, _ = upsampler.enhance(img)
        final_img = cv2.resize(upscaled_img, target_resolution, interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(output_path, final_img)
        end_time = time.time()
        print(f"Successfully upscaled image to {output_path}")
        print(f"Image processing took {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error upscaling image {input_path}: {e}")


def upscale_video(input_path, output_path, upsampler, target_resolution):
    """
    Upscales a video using Real-ESRGAN.
    """
    start_time = time.time()
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return


    with tempfile.TemporaryDirectory() as temp_dir:
        # Check for sufficient disk space
        total, used, free = shutil.disk_usage(temp_dir)
        free_gb = free / (1024**3)
        if free_gb < MIN_REQUIRED_DISK_SPACE_GB:
            print(f"Error: Insufficient disk space in temporary directory ({temp_dir}).")
            print(f"Available: {free_gb:.2f} GB, Required: {MIN_REQUIRED_DISK_SPACE_GB} GB.")
            return

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

    end_time = time.time()
    print(f"Successfully created upscaled video at {output_path}")
    print(f"Video processing took {end_time - start_time:.2f} seconds.")

def process_file(input_file, output_file, args, upsampler):
    input_resolution = get_input_resolution(input_file)
    if input_resolution is None:
        print(f"Skipping {input_file}: Could not determine input resolution.")
        return

    if args.target_resolution:
        if args.target_resolution in RESOLUTION_MAP:
            target_resolution = RESOLUTION_MAP[args.target_resolution]
        else:
            print("Error: Invalid target resolution.")
            return
    else:
        scale_factor = args.scale
        target_resolution = (int(input_resolution[0] * scale_factor), int(input_resolution[1] * scale_factor))

    video_extensions = ['.mp4', '.mkv', '.avi', '.mov']
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    
    input_ext = os.path.splitext(input_file)[1].lower()

    if input_ext in video_extensions:
        upscale_video(input_file, output_file, upsampler, target_resolution)
    elif input_ext in image_extensions:
        upscale_image(input_file, output_file, upsampler, target_resolution)
    else:
        print(f"Skipping {input_file}: Unsupported file format.")

def main():
    parser = argparse.ArgumentParser(description="Upscale a video or image using Real-ESRGAN.")
    parser.add_argument("input_path", help="Path to the input video or image file or directory.")
    parser.add_argument("output_path", help="Path to save the output file or directory.")
    parser.add_argument("--scale", type=float, help="Upscaling factor (e.g., 1.5, 2.0). Mutually exclusive with --target-resolution.")
    parser.add_argument("--target-resolution", type=str, help="Target resolution (e.g., 1080p, 4k). Mutually exclusive with --scale.")
    
    args = parser.parse_args()

    if args.scale and args.target_resolution:
        print("Error: Please provide either --scale or --target-resolution, not both.")
        exit()

    if not args.scale and not args.target_resolution:
        print("Error: Please provide either --scale or --target-resolution.")
        exit()

    if args.scale:
        if args.scale <= 2:
            model_scale = 2
        elif args.scale <= 4:
            model_scale = 4
        else:
            print("Error: Scaling factors greater than 4 are not yet supported.")
            exit()
    else: # args.target_resolution
        if os.path.isdir(args.input_path):
            first_file = next((f for f in os.listdir(args.input_path) if os.path.isfile(os.path.join(args.input_path, f))), None)
            if not first_file:
                print("Error: Input directory is empty.")
                exit()
            input_resolution = get_input_resolution(os.path.join(args.input_path, first_file))
        else:
            input_resolution = get_input_resolution(args.input_path)
        
        if input_resolution is None:
            print("Error: Could not determine input resolution for the first file.")
            exit()
            
        if args.target_resolution in RESOLUTION_MAP:
            target_resolution = RESOLUTION_MAP[args.target_resolution]
            scale_factor = target_resolution[0] / input_resolution[0]
            if scale_factor <= 2:
                model_scale = 2
            elif scale_factor <= 4:
                model_scale = 4
            else:
                print("Error: Scaling factors greater than 4 are not yet supported.")
                exit()
        else:
            print("Error: Invalid target resolution.")
            exit()


    upsampler = initialize_upsampler(model_scale)
    if upsampler is None:
        exit()

    if os.path.isdir(args.input_path):
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        
        for filename in os.listdir(args.input_path):
            input_file = os.path.join(args.input_path, filename)
            output_file = os.path.join(args.output_path, filename)

            if os.path.isfile(input_file):
                process_file(input_file, output_file, args, upsampler)
    else:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        process_file(args.input_path, args.output_path, args, upsampler)


if __name__ == "__main__":
    main()