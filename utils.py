import os
import csv
from pytube import YouTube
import yt_dlp
import re
import subprocess


def remove_file_or_dir(path_:str):
    try:
        os.remove(path_)
    except OSError:
        pass


def write_list_to_csv(data:list, file_path:str):
    with open(file_path, "w") as file:
        csv_writer = csv.writer(file, delimiter="|")
        csv_writer.writerows(data)
    return file_path


def correct_filepath(filepath):
    # replace special characters and spaces with underscores
    path, filename = os.path.split(filepath)
    # filename, ext = os.path.splitext(filename)
    sanitized_filename = re.sub(r'[^\w\s._-]', '_', filename)
    sanitized_filename = re.sub(r'\s+', '_', sanitized_filename)
    #Rename file path
    new_filepath = os.path.join(path, sanitized_filename)
    os.rename(filepath, new_filepath)
    return new_filepath


def download_yt_video(video_url, save_dir="data/videos"):
    video_path = os.path.join(save_dir, f'%(title)s.%(ext)s')
    ydl_opts = {
      'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
      'outtmpl': video_path
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
        info = ydl.extract_info(video_url, download=False)
        output_path = ydl.prepare_filename(info)
        # abs_video_path = ydl.prepare_filename(info)
        # # abs_video_path = os.path.join(save_dir, video_name)
        # ydl.process_info(info)
    # Correct file name
    output_path = correct_filepath(output_path)
    print(f"Success download video from youtube, save at {output_path}")
    return output_path


import subprocess
import os


def extract_sound(video_file_path, save_dir="data/sound", sample_rate=16000):
    file_name = os.path.splitext(os.path.basename(video_file_path))[0] + ".wav"
    out_filepath = os.path.join(save_dir, file_name)
    try:
        os.remove(out_filepath)
    except OSError:
        pass

    cmd = f'ffmpeg -i {video_file_path} -ar {sample_rate} -ac 1 -c:a pcm_s16le {out_filepath} -loglevel warning'
    print(f"{cmd}")
    subprocess.run(cmd.split())
    print(f"Successfully extract sound from video, save at {out_filepath}")
    return out_filepath

def remove_sound(video_file_path, save_dir="data/video_no_sound"):
    file_name = os.path.basename(video_file_path)
    out_filepath = os.path.join(save_dir, file_name)
    try:
        os.remove(out_filepath)
    except OSError:
        pass

    cmd = f"ffmpeg -i {video_file_path} -c:v copy -an {out_filepath} -loglevel warning"
    subprocess.run(cmd.split())
    print(f"Successfully remove sound from video, save at {out_filepath}")
    return out_filepath

