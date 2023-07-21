import os
from PIL import Image
from moviepy.editor import VideoFileClip

def convert_mp4_to_gif(mp4_path, gif_path):
    try:
        clip = VideoFileClip(mp4_path)
        clip.write_gif(gif_path)
        print(f"Converted {mp4_path} to {gif_path}")
    except Exception as e:
        print(f"Failed to convert {mp4_path} to GIF: {e}")

def find_mp4_files(root_folder):
    mp4_files = []
    
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".mp4"):
                mp4_files.append(os.path.join(root, file))
    return mp4_files

def main(root_folder):
    mp4_files = find_mp4_files(root_folder)
    for mp4_file in mp4_files:
        gif_file = mp4_file[:-3] + "gif"
        convert_mp4_to_gif(mp4_file, gif_file)

if __name__ == "__main__":
    # Replace 'path_to_your_folder' with the root folder containing all your folders with MP4 files.
    main("classic/results")
