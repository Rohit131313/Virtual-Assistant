from transformers import pipeline
import torch
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
import sys
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan , SpeechT5Config
from IPython.display import Audio
from datasets import load_dataset
import os
from dotenv import load_dotenv
import sounddevice as sd
import warnings
import google.generativeai as genai
import re

# Suppress specific warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*mel filter has all zero values.*")
warnings.simplefilter(action='ignore', category=FutureWarning)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 1 - Wake word detection
classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", cache_dir="Model",device=device
)
# print(classifier.model.config.id2label)
# print(classifier.model.config.id2label[27]) #marvin
def launch_fn(
    wake_word="marvin",
    prob_threshold=0.5,
    chunk_length_s=2.0,
    stream_chunk_s=0.25,
    debug=False,
):
    if wake_word not in classifier.model.config.label2id.keys():
        raise ValueError(
            f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the set {classifier.model.config.label2id.keys()}."
        )

    sampling_rate = classifier.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s
    )

    print("To Activate Your Virtual Assistant say , Marvin")
    for prediction in classifier(mic):
        prediction = prediction[0]
        if debug:
            print(prediction)
        if prediction["label"] == wake_word:
            if prediction["score"] > prob_threshold:
                return True

# 2 - Speech transcription
transcriber = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base.en",device=device
)
def transcribe(chunk_length_s=5.0, stream_chunk_s=1.0):
    sampling_rate = transcriber.feature_extractor.sampling_rate

    mic = ffmpeg_microphone_live(
        sampling_rate=sampling_rate,
        chunk_length_s=chunk_length_s,
        stream_chunk_s=stream_chunk_s,
    )

    print("How Can I Assist You Today ?")
    for item in transcriber(mic, generate_kwargs={"max_new_tokens": 200}):
        sys.stdout.write("\033[K")
        print(item["text"], end="\r")
        if not item["partial"][0]:
            break

    return item["text"]

# Load the .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 3 - Language model query
def query(text):
    print(f"Querying...: {text}")
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content("Give the summarize answer maximum 100 token of query : "+text)

    return response.text

# 4 - Synthesise speech
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
def synthesise(text):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        inputs["input_ids"].to(device), speaker_embeddings.to(device), vocoder=vocoder
    )
    return speech.cpu()

def play_audio(audio_data, sampling_rate=16000):
    audio_np = audio_data.squeeze().numpy()  # Convert Tensor to numpy array
    sd.play(audio_np, samplerate=sampling_rate)
    sd.wait()  # Wait until audio finishes playing

terminate_text = "\nTo stop the conversation simply say STOP\n"

while launch_fn(debug=False):
    transcription = transcribe()
    if re.sub(r'[^a-z]', '', transcription.lower()).strip() == "stop":
        break
    response = query(transcription)
    print(response)
    audio = synthesise(response)
    play_audio(audio)

    print(terminate_text)


