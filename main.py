# main.py
import argparse
import os
from utils import is_youtube_link, download_yt_video, extract_sound, \
                    remove_sound, change_audio_speed, add_audio_to_video
from speech_to_text import speech_to_text
from translate_to_vietnamese import translate_text_segments
from text_to_speech import translate_audio
from voice_cloning import convert_voice
from moviepy.editor import AudioFileClip

def load_args():
    parser = argparse.ArgumentParser(description="Translate video to Vietnamese arguments")
    parser.add_argument('--video_input', '-v', default="data/videos/eng_male_test_01.mp4", 
                        help='The path of input video or a youtube link', required=False)
    
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
    verify_input_video(video_path)
    if is_youtube_link(video_path):
        video_path = download_yt_video(video_path)
    
    audio_path = extract_sound(video_path)

    org_duration = AudioFileClip(audio_path).duration
    print(f"The duration of the video {video_path} is {org_duration}")

    video_path_no_sound = remove_sound(video_path)

    # transcript_file, _ = speech_to_text(audio_path)

    # translated_transcript_file = translate_text_segments(transcript_file)
    
    # translated_audio_file, _ = translate_audio(translated_transcript_file)

    # converted_audio_file = convert_voice(translated_audio_file)

    converted_audio_file = "data/converted_audio/eng_male_test_01.wav"

    new_duration  = AudioFileClip(converted_audio_file).duration

    speed_factor = new_duration/org_duration + epsilon

    print(f"Speed factor is {speed_factor}")

    aligned_audio_file = change_audio_speed(converted_audio_file, 
                                              out_dir="data/aligned_audio",
                                              speed_factor=speed_factor)


    out_video_path = add_audio_to_video(video_path_no_sound, aligned_audio_file)

    print(f"The output video is {out_video_path}")


if __name__ == "__main__":
    args = load_args()
    main(args)