import torch  # isort:skip

torch.manual_seed(42)
import json
import re
import unicodedata
from types import SimpleNamespace
from scipy.io import wavfile


import gradio as gr
import numpy as np
import regex

from models import DurationNet, SynthesizerTrn

title = "LightSpeed: Vietnamese Male Voice TTS"
description = "Vietnam Male Voice TTS."
config_file = "config.json"
duration_model_path = "vbx_duration_model.pth"
lightspeed_model_path = "gen_619k.pth"
phone_set_file = "vbx_phone_set.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
with open(config_file, "rb") as f:
    hps = json.load(f, object_hook=lambda x: SimpleNamespace(**x))

# load phone set json file
with open(phone_set_file, "r") as f:
    phone_set = json.load(f)

assert phone_set[0][1:-1] == "SEP"
assert "sil" in phone_set
sil_idx = phone_set.index("sil")

space_re = regex.compile(r"\s+")
number_re = regex.compile("([0-9]+)")
digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
num_re = regex.compile(r"([0-9.,]*[0-9])")
alphabet = "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵbcdđghklmnpqrstvx"
keep_text_and_num_re = regex.compile(rf"[^\s{alphabet}.,0-9]")
keep_text_re = regex.compile(rf"[^\s{alphabet}]")


def read_number(num: str) -> str:
    if len(num) == 1:
        return digits[int(num)]
    elif len(num) == 2 and num.isdigit():
        n = int(num)
        end = digits[n % 10]
        if n == 10:
            return "mười"
        if n % 10 == 5:
            end = "lăm"
        if n % 10 == 0:
            return digits[n // 10] + " mươi"
        elif n < 20:
            return "mười " + end
        else:
            if n % 10 == 1:
                end = "mốt"
            return digits[n // 10] + " mươi " + end
    elif len(num) == 3 and num.isdigit():
        n = int(num)
        if n % 100 == 0:
            return digits[n // 100] + " trăm"
        elif num[1] == "0":
            return digits[n // 100] + " trăm lẻ " + digits[n % 100]
        else:
            return digits[n // 100] + " trăm " + read_number(num[1:])
    elif len(num) >= 4 and len(num) <= 6 and num.isdigit():
        n = int(num)
        n1 = n // 1000
        return read_number(str(n1)) + " ngàn " + read_number(num[-3:])
    elif "," in num:
        n1, n2 = num.split(",")
        return read_number(n1) + " phẩy " + read_number(n2)
    elif "." in num:
        parts = num.split(".")
        if len(parts) == 2:
            if parts[1] == "000":
                return read_number(parts[0]) + " ngàn"
            elif parts[1].startswith("00"):
                end = digits[int(parts[1][2:])]
                return read_number(parts[0]) + " ngàn lẻ " + end
            else:
                return read_number(parts[0]) + " ngàn " + read_number(parts[1])
        elif len(parts) == 3:
            return (
                read_number(parts[0])
                + " triệu "
                + read_number(parts[1])
                + " ngàn "
                + read_number(parts[2])
            )
    return num


def text_to_phone_idx(text):
    # lowercase
    text = text.lower()
    # unicode normalize
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    text = text.replace(";", " ; ")
    text = text.replace(":", " : ")
    text = text.replace("!", " ! ")
    text = text.replace("?", " ? ")
    text = text.replace("(", " ( ")

    text = num_re.sub(r" \1 ", text)
    words = text.split()
    words = [read_number(w) if num_re.fullmatch(w) else w for w in words]
    text = " ".join(words)

    # remove redundant spaces
    text = re.sub(r"\s+", " ", text)
    # remove leading and trailing spaces
    text = text.strip()
    # convert words to phone indices
    tokens = []
    for c in text:
        # if c is "," or ".", add <sil> phone
        if c in ":,.!?;(":
            tokens.append(sil_idx)
        elif c in phone_set:
            tokens.append(phone_set.index(c))
        elif c == " ":
            # add <sep> phone
            tokens.append(0)
    if tokens[0] != sil_idx:
        # insert <sil> phone at the beginning
        tokens = [sil_idx, 0] + tokens
    if tokens[-1] != sil_idx:
        tokens = tokens + [0, sil_idx]
    return tokens


def text_to_speech(duration_net, generator, text):
    # prevent too long text
    if len(text) > 500:
        text = text[:500]

    phone_idx = text_to_phone_idx(text)
    batch = {
        "phone_idx": np.array([phone_idx]),
        "phone_length": np.array([len(phone_idx)]),
    }

    # predict phoneme duration
    phone_length = torch.from_numpy(batch["phone_length"].copy()).long().to(device)
    phone_idx = torch.from_numpy(batch["phone_idx"].copy()).long().to(device)
    with torch.inference_mode():
        phone_duration = duration_net(phone_idx, phone_length)[:, :, 0] * 1000
    phone_duration = torch.where(
        phone_idx == sil_idx, torch.clamp_min(phone_duration, 200), phone_duration
    )
    print(f"Phone duration is {phone_duration.shape}")
    # phone_duration = torch.where(phone_idx == 0, 0, phone_duration)
    phone_duration = torch.where(phone_idx == 0, 0, phone_duration)

    # generate waveform
    end_time = torch.cumsum(phone_duration, dim=-1)
    start_time = end_time - phone_duration
    start_frame = start_time / 1000 * hps.data.sampling_rate / hps.data.hop_length
    end_frame = end_time / 1000 * hps.data.sampling_rate / hps.data.hop_length
    spec_length = end_frame.max(dim=-1).values
    pos = torch.arange(0, spec_length.item(), device=device)
    attn = torch.logical_and(
        pos[None, :, None] >= start_frame[:, None, :],
        pos[None, :, None] < end_frame[:, None, :],
    ).float()
    with torch.inference_mode():
        y_hat = generator.infer(
            phone_idx, phone_length, spec_length, attn, max_len=None, noise_scale=0.667
        )[0]
    wave = y_hat[0, 0].data.cpu().numpy()
    return (wave * (2**15)).astype(np.int16)


def load_models():
    duration_net = DurationNet(hps.data.vocab_size, 64, 4).to(device)
    duration_net.load_state_dict(torch.load(duration_model_path, map_location=device))
    duration_net = duration_net.eval()
    generator = SynthesizerTrn(
        hps.data.vocab_size,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **vars(hps.model),
    ).to(device)
    del generator.enc_q
    ckpt = torch.load(lightspeed_model_path, map_location=device)
    params = {}
    for k, v in ckpt["net_g"].items():
        k = k[7:] if k.startswith("module.") else k
        params[k] = v
    generator.load_state_dict(params, strict=False)
    del ckpt, params
    generator = generator.eval()
    return duration_net, generator


def speak(text):
    duration_net, generator = load_models()
    paragraphs = text.split("\n")
    clips = []  # list of audio clips
    # silence = np.zeros(hps.data.sampling_rate // 4)
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph == "":
            continue
        clips.append(text_to_speech(duration_net, generator, paragraph))
        # clips.append(silence)
    y = np.concatenate(clips)
    return hps.data.sampling_rate, y


# gr.Interface(
#     fn=speak,
#     inputs="text",
#     outputs="audio",
#     title=title,
#     examples=[
#         "Trăm năm trong cõi người ta, chữ tài chữ mệnh khéo là ghét nhau.",
#         "Đoạn trường tân thanh, thường được biết đến với cái tên đơn giản là Truyện Kiều, là một truyện thơ của đại thi hào Nguyễn Du",
#         "Lục Vân Tiên quê ở huyện Đông Thành, khôi ngô tuấn tú, tài kiêm văn võ. Nghe tin triều đình mở khoa thi, Vân Tiên từ giã thầy xuống núi đua tài.",
#         "Lê Quý Đôn, tên thuở nhỏ là Lê Danh Phương, là vị quan thời Lê trung hưng, cũng là nhà thơ và được mệnh danh là nhà bác học lớn của Việt Nam trong thời phong kiến",
#         "Tất cả mọi người đều sinh ra có quyền bình đẳng. Tạo hóa cho họ những quyền không ai có thể xâm phạm được; trong những quyền ấy, có quyền được sống, quyền tự do và quyền mưu cầu hạnh phúc.",
#     ],
#     description=description,
#     theme="default",
#     allow_screenshot=False,
#     allow_flagging="never",
# ).launch(debug=False)


if __name__ == "__main__":
    text_ = """Tất cả mọi người đều sinh ra có quyền bình đẳng. Tạo hóa cho họ những quyền không ai có thể xâm phạm được; 
    trong những quyền ấy, có quyền được sống, quyền tự do và quyền mưu cầu hạnh phúc. 
    Tất cả mọi người đều sinh ra có quyền bình đẳng. Tạo hóa cho họ những quyền không ai có thể xâm phạm được; 
    trong những quyền ấy, có quyền được sống, quyền tự do và quyền mưu cầu hạnh phúc. 
    Tất cả mọi người đều sinh ra có quyền bình đẳng. Tạo hóa cho họ những quyền không ai có thể xâm phạm được; 
    trong những quyền ấy, có quyền được sống, quyền tự do và quyền mưu cầu hạnh phúc."""
    text_ =  """bởi vì họ không có hình dạng và đường dễ nhận biết. 
    Do đó, nó phải tích hợp nhiều hình dạng nếu nó phải hiển thị một bàn tay thực tế. 
    Trong các trường hợp khác, trình tạo hình ảnh chỉ bị nhầm lẫn bất cứ khi nào lời nhắc văn bản quá phức tạp. 
    Nó gặp khó khăn khi làm theo hướng dẫn của bạn và xuất ra một hình ảnh méo mó vẫn giống với lời nhắc văn bản nhưng theo cách kỳ cục. 
    Tạo văn bản trong hình ảnh luôn là một khía cạnh khó hiểu của thiết kế công cụ tạo hình ảnh. 
    Mặc dù nghe có vẻ khó khăn, bao gồm nhãn và văn bản trong hình ảnh được tạo đã khiến các nhà phát triển của các mô hình tổng hợp hình ảnh trước đó bị đau nửa đầu nghiêm trọng.
      Dali 3 đã giải quyết được câu đố đó. Với Dali 3, việc thêm chú thích hình ảnh của bạn bây giờ là đi bộ trong công viên.
        Ở một mức độ nào đó, có một cách để giải quyết vấn đề biến dạng, nhưng đó chỉ là nếu bạn biết cách mã hoá. 
        Bạn có thể cải thiện độ trung thực của hình ảnh được tạo bằng cách thực hiện một số kỹ thuật nhắc nhở. 
        Một ví dụ hoàn hảo về mô hình tổng hợp hình ảnh AI yêu cầu hack để tối ưu hoá là Mid Journey. 
        Trình tạo hình ảnh này chỉ có thể tạo ra các chi tiết quang học nếu người dùng tinh chỉnh nhiều cho lời nhắc, 
        nhưng nếu người dùng nghiêm túc sử dụng lời nhắc văn bản như vậy, họ không kiểm soát được đầu ra. 
        Thật không may, chỉ có một số ít người dùng trình tạo hình ảnh có bí quyết kỹ thuật để thực hiện các hack này. 
        Vì vậy, tỷ lệ người dùng lớn hơn phải làm gì? Tìm hiểu các ngôn ngữ lập trình AI? Với Dali 3, 
        bạn không cần phải thực hiện rất nhiều kỹ thuật nhắc nhở phản trực giác. 
        Dali 3 toả sáng trong khả năng tuân theo các hướng dẫn khó khăn và vẫn cung cấp hình ảnh rực rỡ mà không yêu cầu bất kỳ chỉnh sửa bổ sung nào. 
        Như tôi đã đề cập trước đây, Dali 3 được xây dựng trên GPT, 
        vì vậy bạn có thể mượn một số điểm từ trình tạo văn bản nếu bạn cần thêm chi tiết vào tầm nhìn của mình. 
        Chỉ cần nhập một mô tả ngắn về hình ảnh bạn muốn tạo và ChatGPT sẽ cung cấp cho bạn đầu vào tốt nhất có thể để đưa vào lời nhắc của Dali. 
        Ngoài ra, nếu bạn thấy rằng kết quả hình ảnh không bắt được trí tưởng tượng của bạn, 
        bạn có thể yêu cầu ChatGPT giúp cụm từ mô tả văn bản của bạn đúng cách. 
        ChatGPT là tất cả các kỹ thuật nhắc nhở bạn cần. Dali 3 có cùng chính sách bản quyền."""
    sr, y_hat = speak(text_)
    print(f"Sample rate {sr}")
    print(f"Ouput shape is {y_hat.shape}")
    output_file = "test_01.wav"
    wavfile.write(output_file, sr, y_hat)   