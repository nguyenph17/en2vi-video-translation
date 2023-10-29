from faster_whisper import WhisperModel
import os
import csv
import pandas as pd
from googletrans import Translator
from copy import deepcopy
from utils import remove_file_or_dir




def translate_text(text, target_lang_code="vi", src_lang_code="en"):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_lang_code).text
    return translated_text


def merge_consecutive_segments(timestamp_segments_file, epsilon=0.05, out_dir="data/merged_texts/"):
    timestamp_segments = pd.read_csv(timestamp_segments_file,
                                 names=["start", "end", "text"])
    n_segments = len(timestamp_segments)
    merged_segments = []
    current_row = timestamp_segments.iloc[0].tolist() #loc[[0]].tolist()
    for i in range(0, n_segments -1):
        next_row = timestamp_segments.iloc[i+1].tolist()#loc[[i+1]].tolist()
        # if start time of next row - end time of previous row < epsilon -> merge it
        if float(next_row[0]) - float(current_row[1]) <= epsilon:
            current_row[1] = next_row[1]
            current_row[2] += next_row[2]
        else:
            merged_segments.append(current_row)
            current_row = next_row
    # concat the last row to the list
    merged_segments.append(current_row)

    file_name = os.path.basename(timestamp_segments_file)
    file_path = os.path.join(out_dir, file_name)
    remove_file_or_dir(file_path)
    with open(file_path, "w") as file:
        csv_writer = csv.writer(file, delimiter=",")
        csv_writer.writerows(merged_segments)
    return file_path


# Load text from csv
def create_translated_speech_csv(speech_csv_filepath, out_dir="data/translated_texts"):
    filename = os.path.basename(speech_csv_filepath)
    full_text = ""
    timestamp_segments = pd.read_csv(speech_csv_filepath,
                                 names=["start", "end", "text"])
    print(f"There are {len(timestamp_segments)} segments before translating!")
    # check if there are any empty string
    """
    empty_row_idx = timestamp_text[timestamp_text['text'] == ''].index
    if len(empty_row_idx) > 0:
        print(f"There ")
    """
    # get full text
    full_text = '|'.join(timestamp_segments["text"][:10])
    print(f"There are {len(full_text.split())} words.")
    # translate the text
    trans_full_text = translate_text(full_text)
    print(f"Translated text: {trans_full_text}")
    translated_text_segments = trans_full_text.split("|")
    print(f"There are {len(translated_text_segments)} segments after translating!")
    # timestamp_segments['text'] = translated_text_segments
    # save thte translated timestamp speech textt
    # timestamp_segments.to_csv(os.path.join(out_dir, filename))
    # Writing to CSV file
    return timestamp_segments