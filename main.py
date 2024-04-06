# Copyright 2024 Gokul J
# Licensed under the MIT License.
# This project is available at https://github.com/gokulj2005/Voice-Translation-With-Cloning.git


import argparse
import os
from pathlib import Path
import time
import librosa
import numpy as np
import soundfile as sf
import torch

import gradio as gr
from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder
import subprocess




def update_translation(input_lang, output_lang):
    global translated_text
    audio_file = "input_audio.wav"
    model = "small"
    cmd = f"whisper {audio_file} --language {input_lang} --task translate --model {model}"
    translated_text = subprocess.check_output(cmd, shell=True, text=True)
    translated_text = translated_text[27:]
    translated_text = translated_text.strip()
    print(translated_text)
    return translated_text

enc_model_fpath="saved_models/default/encoder.pt"
syn_model_fpath="saved_models/default/synthesizer.pt"
voc_model_fpath="saved_models/default/vocoder.pt"
seed = None
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-e", "--enc_model_fpath", type=Path,
                    default=enc_model_fpath,
                    help="Path to a saved encoder")
parser.add_argument("-s", "--syn_model_fpath", type=Path,
                    default=syn_model_fpath,
                    help="Path to a saved synthesizer")
parser.add_argument("-v", "--voc_model_fpath", type=Path,
                    default=voc_model_fpath,
                    help="Path to a saved vocoder")
parser.add_argument("--seed", type=int, default=seed, help=\
    "Optional random number seed value to make toolbox deterministic.")
    
args = parser.parse_args()
arg_dict = vars(args)
print_args(args, parser)
    

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
        "%.1fGb total memory.\n" %
        (torch.cuda.device_count(),
        device_id,
        gpu_properties.name,
        gpu_properties.major,
        gpu_properties.minor,
        gpu_properties.total_memory / 1e9))
else:
    print("Using CPU for inference.\n")

print("Preparing the encoder, the synthesizer and the vocoder...")
ensure_default_models(Path("saved_models"))
encoder.load_model(args.enc_model_fpath)
synthesizer = Synthesizer(args.syn_model_fpath)
vocoder.load_model(args.voc_model_fpath)

    
    
    





def generate_voice(text):
    global encoder, synthesizer, vocoder
    try:
            
        in_fpath = "input_audio.wav"

        preprocessed_wav = encoder.preprocess_wav(in_fpath)
        original_wav, sampling_rate = librosa.load(str(in_fpath))
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        print("Loaded file succesfully")

        embed = encoder.embed_utterance(preprocessed_wav)
        print("Created the embedding")



        if args.seed is not None:
            torch.manual_seed(args.seed)
            synthesizer = Synthesizer(args.syn_model_fpath)
        
        texts = [text]
        embeds = [embed]
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        print("Created the mel spectrogram")

        print("Synthesizing the waveform:")

        if args.seed is not None:
            torch.manual_seed(args.seed)
            vocoder.load_model(args.voc_model_fpath)

        generated_wav = vocoder.infer_waveform(spec)

        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        generated_wav = encoder.preprocess_wav(generated_wav)

        filename = "output_audio.wav"
        print(generated_wav.dtype)
        sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
        
        print("\nSaved output as %s\n\n" % filename)
        
    except Exception as e:
        print("Caught exception: %s" % repr(e))
        print("Restarting\n")

def run_g(input_lang,output_lang,audio_file):
    if os.path.exists("input_audio.wav"):
        os.remove("input_audio.wav")
        print("input_audio.wav removed")

    if os.path.exists("output_audio.wav"):
        os.remove("output_audio.wav")
        print("output_audio.wav removed")
    audio_data, sample_rate = sf.read(audio_file)
    sf.write('input_audio.wav',audio_data, sample_rate)
    
    translated_text = update_translation(input_lang,output_lang)
    if translated_text == "Thank you for watching" or translated_text == "":
        return "output_audio.wav", "Could not understand! Please try again."
    else:
        generate_voice(translated_text)
        return "output_audio.wav", translated_text 
        

    


demo = gr.Interface(
    fn=run_g,
    inputs = [gr.Dropdown(
            ["Tamil", "Malayalam", "Hindi","Kannada","Telugu"], label="Language", value = "Tamil", interactive = True
        ), gr.Dropdown(
            ["English"], label="Language" , value = "English", interactive = True
        ), gr.Audio(sources="microphone", type="filepath")],
     outputs=[gr.Audio(type="filepath", label="Translated Audio", autoplay=True),gr.Textbox(label="Translated Text")],
    title="Voice Translation with Cloning",
    description="Translate speech from Tamil, Malayalam, Hindi, Kannada, Telugu to English and synthesize the translated text into speech. This project is available at <a href='https://github.com/gokulj2005/Voice-Translation-With-Cloning.git' target='_blank'>https://github.com/gokulj2005/Voice-Translation-With-Cloning.git</a>"
).launch()
