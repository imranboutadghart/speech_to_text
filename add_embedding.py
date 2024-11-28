#!/home/iboutadg/tests/se_project/se/bin/python3
from pyannote.audio import Inference
import librosa
import torch
import sys
import os

import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
from get_audio import save_audio
auth_token = ""

def denoise(input_file, output_file):
    # outputs denoised file
    model = pretrained.dns64().cuda()
    wav, sr = torchaudio.load(input_file)
    wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised = model(wav[None])[0]
    # save audio after displaying
    denoised_cpu = denoised.cpu()
    torchaudio.save(output_file, denoised_cpu, model.sample_rate)
    return output_file

def main():
    the_audio = save_audio()
    print("denoising " + the_audio)
    tmp_dir = "tmp/"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    denoised_output = denoise(the_audio, tmp_dir + the_audio + "_denoised.wav")
    audio_file = denoised_output
    embedding_dir = "embeddings/"
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)
    tmp = audio_file.split("/")[-1]
    output_file = embedding_dir + tmp + "_embedding.pt"
    speaker_embedding_model = Inference("pyannote/embedding", use_auth_token=auth_token)
    audio, sr = librosa.load(audio_file, sr=16000)
    audio_tensor = torch.tensor(audio).unsqueeze(0)
    embedding = speaker_embedding_model({'waveform': audio_tensor, 'sample_rate': 16000})
    torch.save(embedding, output_file)
    print("embedding was saved as " + output_file)

if __name__ == "__main__":
    main()
