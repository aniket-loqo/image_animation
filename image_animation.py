import os
import shutil
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, AudioFileClip
from pydub import AudioSegment
from pydub.audio_segment import AudioSegment


from moviepy.editor import AudioClip
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np

def create_silent_audio(duration, sample_rate=44100):
    """Create a silent audio clip for the specified duration in seconds."""
    # Create a silent array (all zeros) with the specified duration and sample rate
    silent_audio_array = np.zeros(int(duration * sample_rate))
    silent_audio_clip = AudioArrayClip([silent_audio_array], fps=sample_rate)
    return silent_audio_clip.set_duration(duration)

from moviepy.editor import CompositeVideoClip

def slide_transition(clip1, clip2, duration=0.5):
    """
    Creates a sliding transition between two clips.
    Clip1 slides out to the left, and Clip2 slides in from the right.
    """
    # Clip1 slides out to the left
    clip1_slide_out = clip1.set_position(
        lambda t: ('center', 'center') if t <= 0 else (-clip1.w * (t / duration), 'center')).set_duration(duration)

    # Clip2 slides in from the right
    clip2_slide_in = clip2.set_position(
        lambda t: (clip2.w * (1 - t / duration), 'center') if t <= duration else ('center', 'center')).set_duration(
        duration)

    # Composite the two clips to have the transition effect
    return CompositeVideoClip([clip1_slide_out, clip2_slide_in.set_start(duration)])

def shake_effect2(image1, image2, duration=15, speed=20, strength=10, noise_intensity=0.1):
    """Apply a shake effect between two images."""
    result = []
    height, width, _ = image1.shape
    for i in range(duration):
        x_shift = int(np.sin(i / speed) * strength * (1 + np.random.randn() * noise_intensity))
        y_shift = int(np.cos(i / speed) * strength * (1 + np.random.randn() * noise_intensity))
        x_shift = np.clip(x_shift, -strength, strength)
        y_shift = np.clip(y_shift, -strength, strength)
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        shifted = cv2.warpAffine(image1, M, (width, height))
        result.append(cv2.addWeighted(shifted, 1, image2, 0, 0))
    return result

# transition effect 1
def zoom_effect2(image1, image2, duration=90, zoom_strength=1.5, transition_type='out'):
    result = []
    height, width, _ = image1.shape

    for i in range(duration):
        if transition_type == 'in':
            zoom_factor = 1 + (zoom_strength - 1) * i / duration
        elif transition_type == 'out':
            zoom_factor = zoom_strength - (zoom_strength - 1) * i / duration
        else:
            raise ValueError("Invalid transition type. Choose 'in' or 'out'.")

        resized = cv2.resize(image1, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

        if zoom_factor > 1:
            # Crop
            cropped = resized[:height, :width]
        else:
            # Pad
            pad_width = int((width - resized.shape[1]) / 2)
            pad_height = int((height - resized.shape[0]) / 2)
            cropped = cv2.copyMakeBorder(resized, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT)

        result.append(cv2.addWeighted(cropped, 1, image2, 0, 0))

    return result

from moviepy.editor import ImageSequenceClip

def whip_pan_effect(img1, img2, steps=30, direction='right'):
    h, w = img1.shape[:2]
    if img1.shape[:2] != img2.shape[:2]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    frames = []
    for step in range(steps):
        offset = int((step / steps) * w)
        if direction == 'right':
            pan_frame = np.zeros_like(img1)
            pan_frame[:, :w - offset] = img1[:, offset:]
            pan_frame[:, w - offset:] = img2[:, :offset]
        elif direction == 'left':
            pan_frame = np.zeros_like(img1)
            pan_frame[:, offset:] = img1[:, :w - offset]
            pan_frame[:, :offset] = img2[:, w - offset:]
        frames.append(pan_frame)
    return frames

# Dissolve effect
def dissolve_effect(img1, img2, steps=30):
    """Apply dissolve transition between two images."""
    frames = []
    for alpha in np.linspace(0, 1, steps):
        dissolved_frame = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        frames.append(dissolved_frame)
    return frames

# Modified apply_transition_effect function to include dissolve effect
def apply_transition_effect(video_clips, effect="dissolve", duration=15, **kwargs):
    """Apply a specified transition effect between each consecutive video clip."""
    final_clips = []
    for i in range(len(video_clips) - 1):
        current_clip = video_clips[i]
        next_clip = video_clips[i + 1]
        
        end_frame = current_clip.get_frame(current_clip.duration - (1 / current_clip.fps))
        start_frame = next_clip.get_frame(0)

        # Apply specified transition effect
        if effect == "dissolve":
            transition_frames = dissolve_effect(end_frame, start_frame, steps=duration)
        elif effect == "shake":
            transition_frames = shake_effect2(end_frame, start_frame, duration=duration)
        elif effect == "whip_pan":
            direction = kwargs.get("direction", "right")
            transition_frames = whip_pan_effect(end_frame, start_frame, steps=duration, direction=direction)
        else:
            raise ValueError("Unsupported transition effect type. Use 'dissolve', 'shake', or 'whip_pan'.")

        # Convert frames to ImageSequenceClip
        transition_clip = ImageSequenceClip(transition_frames, fps=current_clip.fps)
        final_clips.extend([current_clip, transition_clip])
    
    # Append the last clip
    final_clips.append(video_clips[-1])
    return concatenate_videoclips(final_clips)

def animated_video(folder_structure, output_video_path, videos_folder, final_audio_file_path=None,
                   bg_extra_audio_file_path=None, clip_durations=None, target_duration=40):
    # Verify videos folder exists
    if not os.path.exists(videos_folder):
        raise FileNotFoundError(f"Videos folder does not exist: {videos_folder}")

    # List available video files
    video_files = [vid for vid in os.listdir(videos_folder) if vid.lower().endswith(('.mp4', '.mov', '.avi'))]
    print(f"Available video files in '{videos_folder}': {video_files}")
    video_files.sort()

    if len(video_files) == 0:
        raise ValueError("No video files found in the specified folder.")

    # Adjust clip durations if needed
    if len(clip_durations) < len(video_files):
        clip_durations = (clip_durations * ((len(video_files) // len(clip_durations)) + 1))[:len(video_files)]
    elif len(clip_durations) > len(video_files):
        clip_durations = clip_durations[:len(video_files)]

    video_paths = [os.path.normpath(os.path.join(videos_folder, vid)) for vid in video_files]
    video_clips = []

    # Load each video clip and adjust duration if necessary
    for i, video_path in enumerate(video_paths):
        print(f"Loading video [{i}]: {video_path}")
        try:
            clip = VideoFileClip(video_path)
            target_clip_duration = clip_durations[i] if clip_durations else clip.duration
            if clip.duration != target_clip_duration:
                speed_factor = clip.duration / target_clip_duration
                clip = clip.fx(vfx.speedx, speed_factor).set_duration(target_clip_duration)
            video_clips.append(clip)
            print(f"Loaded video [{i}] successfully with duration: {clip.duration}")
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            raise ValueError(f"Error loading video: {video_path}")

    # Apply slide effect transition between clips
    try:
        final_video_clip = apply_transition_effect(video_clips, effect="dissolve", duration=15)


    except Exception as e:
        print(f"Error applying transition effect: {e}")
        raise ValueError("Error applying transition effect.")

    # Load or create audio track
    final_audio_clip = None
    if final_audio_file_path and os.path.exists(final_audio_file_path):
        try:
            audio_clip = AudioFileClip(final_audio_file_path).subclip(0, final_video_clip.duration)
            final_audio_clip = audio_clip
        except Exception as e:
            print(f"Warning: Could not load main audio: {e}")

    # If no audio, create silent audio
    if not final_audio_clip:
        print("No audio found, adding silent audio track.")
        final_audio_clip = create_silent_audio(final_video_clip.duration)

    # Set audio to final concatenated video clip
    final_video_clip = final_video_clip.set_audio(final_audio_clip)
    print(f"Writing final video to: {output_video_path}")

    try:
        final_video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac', fps=24)
    except Exception as e:
        print(f"Error writing video file: {e}")
        raise ValueError("Error writing video file.")

    # Cleanup
    final_video_clip.close()
    if final_audio_clip:
        final_audio_clip.close()



def close_processes():
    # Helper function to close any open processes
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name']):
            if 'ffmpeg' in proc.info['name']:
                proc.kill()
    except (ImportError, psutil.NoSuchProcess):
        pass

def main():
    # Folder structure class
    class FolderStructure:
        output_audio_path = "./output_audio"
    
    # Set paths and parameters
    output_video_path = "./output_video.mp4"
    final_audio_file_path = r"C:\temp_video\background-music.mp3"
    bg_extra_audio_file_path = r"C:\temp_video\sound-effects-end.mp3"
    videos_folder = r"C:\temp_video\animation"
    clip_durations = [2, 3, 3.5, 1.5, 4, 3, 3]

    # Debugging paths
    print("Checking if paths exist:")
    print(f"Animation Folder Exists: {os.path.exists(videos_folder)}")
    print(f"Audio File Exists: {os.path.exists(final_audio_file_path)}")
    print(f"Background Audio File Exists: {os.path.exists(bg_extra_audio_file_path)}")

    # Run animated video generation
    try:
        animated_video(FolderStructure(), output_video_path, videos_folder, 
                       final_audio_file_path, bg_extra_audio_file_path, clip_durations, target_duration=40)
        print("Video generated successfully!")
    except (FileNotFoundError, ValueError) as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()