from faster_whisper import WhisperModel
import os
import csv
import pandas as pd
from googletrans import Translator
from copy import deepcopy
from utils import remove_file_or_dir


def export_transcript_to_file(segments, out_filepath):
    remove_file_or_dir(out_filepath)
    with open(out_filepath, "w+") as f:
        csv_writer = csv.writer(f, delimiter="|")
        for segment in segments:
            csv_writer.writerow([segment.start, segment.end, segment.text])
    return out_filepath


def speech_to_text(audio_file, language=None, out_dir="data/transcripts"):
    # Run on GPU with FP16
    model_size = "medium"
    # model = WhisperModel(model_size, device="cpu", compute_type="int8", download_root="model", local_files_only=True)
    print(f"Using Whisper model version {model_size}")
    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8", download_root="models", local_files_only=True)
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        best_of=5,
        language=language,
        # no_speech_threshold=0.5,
        # vad_filter=True,
        # vad_parameters=dict(min_silence_duration_ms=500),
        # word_timestamps=True,
    )

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    filename, ext = os.path.splitext(os.path.basename(f"{audio_file}"))
    output_file = os.path.join(out_dir, filename + ".csv")
    output_file = export_transcript_to_file(segments, output_file)
    print(f"Output file saved at {output_file}")
    return output_file, info