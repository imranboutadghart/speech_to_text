import sys
import os

import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

def denoise(input_file, output_file):
    # outputs denoised file
    print("denoising " + input_file)
    model = pretrained.dns64().cuda()
    wav, sr = torchaudio.load(input_file)
    wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised = model(wav[None])[0]
    # save audio after displaying
    denoised_cpu = denoised.cpu()
    torchaudio.save(output_file, denoised_cpu, model.sample_rate)
    return output_file

def make_dirs():
    if not os.path.exists("tmp/"):
        os.makedirs("tmp/")
    if not os.path.exists("embeddings/"):
        os.makedirs("embeddings/")
    if not os.path.exists("output/"):
        os.makedirs("output/")
