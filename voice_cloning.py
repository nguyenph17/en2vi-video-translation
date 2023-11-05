# voice_cloning.py

import os
import subprocess

# config_file = "models/voice_clone/config.json"
# model_file = "models/voice_clone/G_415.pth"
# input_file = "data/sound/test_video_2_chunk_000.wav"
# output_dir = "data/translated_audio/test_video_2_chunk_000.wav"

# !svc infer {input_file} -m {model_file} -c{config_file} -o {output_dir}

def convert_voice(input_audio, 
                    out_dir="data/converted_audio",
                    model_file="models/voice_clone/G_415.pth", 
                    config_file="models/voice_clone/config.json"):
    
    file_name = os.path.basename(input_audio)
    output_path = os.path.join(out_dir, file_name)

    cmd = f"svc infer {input_audio} -m {model_file} -c{config_file} -o {output_path}"
    subprocess.run(cmd.split())
    print(f"Completed converting voice. File save at {output_path}")
    return output_path


if __name__ == "__main__":
    input_file = "data/audio/test_video_2_chunk_000.wav"
    convert_voice(input_audio=input_file,
                    out_dir="data/translated_transcripts",
                    model_file = "models/voice_clone/G_415.pth",
                    config_file = "models/voice_clone/config.json")