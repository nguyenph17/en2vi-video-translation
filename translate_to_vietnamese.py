import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import csv
from utils import write_list_to_csv
import os

tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi", src_lang="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi")

def translate_en2vi(en_text: str) -> str:
    """Translate English text to Vietnamese

    Args:
        en_text (str): input text with the limit to 3000 CHARACTERS

    Returns:
        str: Vietnamese text
    """
    input_ids = tokenizer_en2vi(en_text, return_tensors="pt").input_ids
    output_ids = model_en2vi.generate(
        input_ids,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    vi_text = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    vi_text = " ".join(vi_text)
    return vi_text



def translate_text_segments(text_segment_file, out_dir='data/translated_texts/'):
    with open(text_segment_file) as csv_file:
        translated_segments = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            translated_text = translate_en2vi(row[2])
            print(translated_text)
            row[2] = translated_text
            translated_segments.append(row)
    
    file_name = os.path.basename(text_segment_file)
    file_path = os.path.join(out_dir, file_name)
    return write_list_to_csv(translated_segments, file_path)