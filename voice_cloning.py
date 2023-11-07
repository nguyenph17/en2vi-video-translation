# voice_cloning.py
import os
import subprocess


def convert_voice(input_audio, 
                    out_dir="data/converted_audio",
                    model_file="models/voice_clone/G_415.pth", 
                    config_file="models/voice_clone/config.json"):
    """
    using so-vits-svc-fork to trained the model with your desired voice
    To train the model you can refer https://github.com/voicepaw/so-vits-svc-fork
    """    
    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.basename(input_audio)
    output_path = os.path.join(out_dir, file_name)

    cmd = f"svc infer {input_audio} -m {model_file} -c{config_file} -o {output_path}"
    subprocess.run(cmd.split())
    return output_path
