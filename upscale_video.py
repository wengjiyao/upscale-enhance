

import os
import subprocess
import tempfile
import argparse
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

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


def upscale_video(input_path, output_path, scale=4):
    """
    Upscales a video using Real-ESRGAN.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    if scale == 4:
        model_path = os.path.join(os.path.expanduser("~"), ".cache/realesrgan/RealESRGAN_x4plus.pth")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    elif scale == 2:
        model_path = os.path.join(os.path.expanduser("~"), ".cache/realesrgan/RealESRGAN_x2plus.pth")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    else:
        print("Error: Invalid scale factor. Please choose 2 or 4.")
        return

    print(f"Initializing Real-ESRGAN model for {scale}x upscaling...")
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True, # Use half precision for speed
        gpu_id=0   # Use the first GPU
    )

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
            except Exception as e:
                print(f"Error upscaling frame {frame_name}: {e}")
                return # Stop processing
            
            final_img = cv2.resize(upscaled_img, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)

            save_path = os.path.join(upscaled_frames_dir, frame_name)
            cv2.imwrite(save_path, final_img)
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
    parser = argparse.ArgumentParser(description="Upscale a video using Real-ESRGAN.")
    parser.add_argument("input_video", help="Path to the input video file.")
    parser.add_argument("output_video", help="Path to save the output video file.")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 4], help="Upscaling factor (2 or 4). Default is 4.")
    
    args = parser.parse_args()
    
    upscale_video(args.input_video, args.output_video, args.scale)
