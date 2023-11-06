# main.py
import argparse
import os
from utils import is_youtube_link, download_yt_video, extract_sound, \
                    remove_sound, change_audio_speed, add_audio_to_video
from speech_to_text import speech_to_text
from translate_to_vietnamese import translate_text_segments
from text_to_speech import generate_speech
from voice_cloning import convert_voice
from moviepy.editor import AudioFileClip

def load_args():
    parser = argparse.ArgumentParser(description="Translate video to Vietnamese arguments")
    parser.add_argument('--video_input', '-v', default="data/videos/eng_male_test_01.mp4", 
                        help='The path of input video or a youtube link', required=False)
    
    parser.add_argument('--device', '-d', default="cpu", 
                        help='device to run models gpu or cpu')
    
    parser.add_argument('--epsilon', '-e', default=0.01, 
                        help='The value to control the play speed')
    args = parser.parse_args()
    return args


def verify_input_video(video_path):
    if os.path.isfile(video_path) or is_youtube_link(video_path):
        return video_path
    else:
        raise FileNotFoundError(f"Cannot find the file {video_path}! Please input the video path or youtube link.")


def main(args):
    video_path = args.video_input
    epsilon = args.epsilon
    # Check if the video_path is a video file path or youtube link
    verify_input_video(video_path)
    if is_youtube_link(video_path):
        # Download youtube video
        video_path = download_yt_video(video_path)
    
    # Extract audio from mp4 file
    audio_path = extract_sound(video_path)
    # Get the duration
    org_duration = AudioFileClip(audio_path).duration
    print(f"The duration of the video {video_path} is {org_duration}")
    # Create video without audio
    video_path_no_sound = remove_sound(video_path)
    # Get transcript from audio
    print(f"Step 1: Extracting transcript from audio.................")
    transcript_file, _ = speech_to_text(audio_path, device=args.device)
    print(f"The transcript file is saved at {transcript_file}")
    # Translate the transcript, we can use GPT API to translate to Vietnamese
    print(f"Step 2: Translating transcript.................")
    translated_transcript_file = translate_text_segments(transcript_file)
    print(f"The translated transcript file is saved at {translated_transcript_file}")
    # text to speech with default voice
    print(f"Step 3: Generating speech from transcript.................")
    translated_audio_file, _ = generate_speech(translated_transcript_file)
    print(f"The translated audio file is saved at {translated_audio_file}")
    # Convert the default voice to your own voice (which has been trained)
    print(f"Step 4: Converting voice.................")
    converted_audio_file = convert_voice(translated_audio_file)
    print(f"The audio with desired voice file is saved at {converted_audio_file}")

    # Align the length of new audio with original audio
    print(f"Step 5: Aligning the audio length.................")
    new_duration  = AudioFileClip(converted_audio_file).duration
    # Get speed factor to align audio
    speed_factor = new_duration/org_duration + epsilon
    print(f"Speed factor is {speed_factor}")
    aligned_audio_file = change_audio_speed(converted_audio_file, 
                                              out_dir="data/aligned_audio",
                                              speed_factor=speed_factor)

    print(f"The aligned audio file is saved at {aligned_audio_file}")
    # Combine the new audio and video
    print(f"Step 6: Generating output video.................")
    out_video_path = add_audio_to_video(video_path_no_sound, aligned_audio_file)
    print(f"The output video is {out_video_path}")


if __name__ == "__main__":
    args = load_args()
    main(args)