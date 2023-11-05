import os
import csv
import yt_dlp
import re
import subprocess
import pyrubberband as pyrb
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip


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


def extract_sound(video_file_path, save_dir="data/audio", sample_rate=16000):
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


def is_youtube_link(link):
    # Regular expression pattern to match YouTube video URLs
    youtube_url_pattern = r'^https?://(?:www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]+|https?://youtu\.be/[A-Za-z0-9_-]+'
    
    # Use the re.match function to check if the link matches the pattern
    match = re.match(youtube_url_pattern, link)
    
    # If there's a match, it's a YouTube video link; otherwise, it's not
    return bool(match)


# def change_audio_speed(input_audio, out_dir, speed_factor):
#     # Run the ffmpeg command to change audio speed
#     file_name = os.path.basename(input_audio)
#     output_file = os.path.join(out_dir, file_name)

#     sample_rate, audio_data = wavfile.read(input_audio)
#     new_sample_rate = int(sample_rate / speed_factor)

#     resampled_audio = resample(audio_data, int(len(audio_data) * new_sample_rate / sample_rate))
                               
#     wavfile.write(output_file, new_sample_rate, resampled_audio)
#     return output_file, new_sample_rate



def change_audio_speed(input_audio, out_dir, speed_factor):
    try:
        # Load the audio file
        audio = AudioSegment.from_file(input_audio)

        file_name = os.path.basename(input_audio)
        output_file = os.path.join(out_dir, file_name)

        # Adjust the speed while preserving pitch
        adjusted_audio = audio.speedup(playback_speed=speed_factor)

        # Export the adjusted audio
        adjusted_audio.export(output_file, format="wav")

        return output_file
    except Exception as e:
        return None


def add_audio_to_video(video_path, audio_path, out_dir="outputs"):
    try:
        video_name = os.path.basename(video_path)
        output_path = os.path.join(out_dir, video_name)

        # Step 1: Load the video and audio files
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)

        # Step 2: Extract a subclip of the video to match the duration of the audio
        # If the video is longer than the audio, this step is not necessary.
        # video = ffmpeg_extract_subclip(video_path, 0, audio.duration)

        # Step 3: Set the audio of the video to the loaded audio
        video = video.set_audio(audio)

        # Step 4: Save the video with the new audio
        video.write_videofile(output_path, codec="libx264")

        # Step 5: Close the audio and video objects
        video.close()
        audio.close()
        return output_path
    except Exception as e:
        raise Exception(f"An error occurred: {e}")
