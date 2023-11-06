# Video Language Conversion Project

This project aims to convert a video from one language to another. It involves several modules that perform various tasks such as extracting the transcript from the video, translating the transcript to Vietnamese, creating speech from the translated transcript, and converting the speech to a desired voice using voice cloning techniques.

## Modules

### 1. Get transcript from video (Whisper - speech to text)

This module utilizes the [Whisper Speech-to-Text (STT)](https://github.com/openai/whisper/tree/main) system to extract the transcript from the video.

### 2. Translate the transcript to Vietnamese

This module focuses on translating the extracted transcript to Vietnamese. The model used is [VinAI Translator en2vi](https://huggingface.co/vinai/vinai-translate-en2vi). You can replace this module by ChatGPT API for the better translation.

### 3. Create speech from translated transcript (text to speech)

This module converts the translated transcript into speech using a Text-to-Speech (TTS) system. In this project I use [LightSpeed: Vietnamese Male Voice TTS](https://github.com/NTT123/light-speed)

Input the translated transcript, and the module will generate speech audio files corresponding to the translated text.

### 4. Convert to desired voice (voice cloning)

This module focuses on converting the generated speech to a desired voice using voice cloning techniques. Voice cloning involves training a model with a target voice and then using the model to generate speech in that voice.

In this module, [the desired voice](https://www.youtube.com/@ThuyUyenSachNoi) had been fine-tuned on the model [so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork). You can also fine-tune your own voice.
