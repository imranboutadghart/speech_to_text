#!/home/iboutadg/tests/se_project/se/bin/python3
from pyannote.audio import Inference
import librosa
import torch
import sys
import os
from get_audio import save_audio
from my_utils import denoise, make_dirs

auth_token = ""

def main():
    if (len(sys.argv) < 2):
        the_audio = save_audio()
    else:
        the_audio = sys.argv[1]
    make_dirs()
    tmp_dir = "tmp/"
    denoised_output = denoise(the_audio, tmp_dir + os.path.basename(the_audio) + "_denoised.wav")
    audio_file = denoised_output
    embedding_dir = "embeddings/"
    tmp = os.path.basename(audio_file)
    output_file = embedding_dir + tmp + "_embedding.pt"
    speaker_embedding_model = Inference("pyannote/embedding", use_auth_token=auth_token)
    audio, sr = librosa.load(audio_file, sr=16000)
    audio_tensor = torch.tensor(audio).unsqueeze(0)
    embedding = speaker_embedding_model({'waveform': audio_tensor, 'sample_rate': 16000})
    torch.save(embedding, output_file)
    print("embedding was saved as " + output_file)

if __name__ == "__main__":
    main()
