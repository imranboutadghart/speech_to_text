#!/home/iboutadg/tests/se_project/se/bin/python3
import os
import sys
from googletrans import Translator
import whisper
from IPython import display as disp
import torchaudio
from pyannote.audio import Inference
import torch
import librosa
from sklearn.cluster import KMeans
import numpy as np
from pyannote.audio.pipelines import SpeakerDiarization
import soundfile as sf
import json
from my_utils import denoise, make_dirs

config = json.load(open("config.json"))
auth_token = ""
enable_diarization=config["enable_diarization"]
enable_translation=config["enable_translation"]
language=config["language"]

def transcribe_with_external_translation(audio_path, out_lang=language, model_name='base'):
    # Load the Whisper model
    print("loading whisper model" + model_name)
    model = whisper.load_model(model_name)

    # Load and preprocess the audio
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"Detected language: {detected_language}")

    # Step 1: Transcribe the audio in the original language
    original_options = whisper.DecodingOptions(language=detected_language)
    original_result = whisper.decode(model, mel, original_options)
    original_text = original_result.text
    print("Original text:", original_text)

    # Step 2: Translate the original text to the target language using Google Translate
    if (enable_translation):
        translator = Translator()
        translated_text = translator.translate(original_text, dest=out_lang).text
    else:
        translated_text = original_text
    print(f"Translated text ({out_lang}):", translated_text)

    return original_text, translated_text

def load_models(auth_token):
    print("Loading models speaker embedding + diarization")
    speaker_embedding_model = Inference("pyannote/embedding", use_auth_token=auth_token)
    diarization_pipeline = SpeakerDiarization.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=auth_token
    )
    return speaker_embedding_model, diarization_pipeline

# Fonction pour effectuer la diarisation
def perform_diarization(diarization_pipeline, audio_file):
    diarization = diarization_pipeline({'uri': 'filename', 'audio': audio_file})
    return diarization

# Fonction pour compter les locuteurs uniques
def count_speakers(diarization):
    unique_speakers = set()
    for _, _, speaker in diarization.itertracks(yield_label=True):
        unique_speakers.add(speaker)
    return len(unique_speakers)

# Fonction pour extraire les embeddings des segments audio
def extract_embeddings(diarization, audio_file, speaker_embedding_model):
    embeddings = []
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = turn.start
        end_time = turn.end
        audio_segment, sr = librosa.load(audio_file, sr=16000, offset=start_time, duration=end_time-start_time)
        audio_tensor = torch.tensor(audio_segment).unsqueeze(0)
        embedding = speaker_embedding_model({'waveform': audio_tensor, 'sample_rate': 16000})
        if hasattr(embedding, 'data'):
            embedding_tensor = torch.tensor(embedding.data).mean(dim=0)
        else:
            embedding_tensor = embedding
        embeddings.append(embedding_tensor.detach().numpy())
        segments.append((start_time, end_time))
    return np.vstack(embeddings).astype(np.float64), segments

# Fonction pour appliquer K-Means clustering
def apply_kmeans(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans

# Fonction pour identifier les segments correspondant au professeur
def identify_professor_segments(embeddings, segments, kmeans, audio_file, professor_embedding_tensor):
    segments_professor = []
    for i, embedding_tensor in enumerate(embeddings):
        embedding_tensor = embedding_tensor.reshape(1, -1)
        cluster_label = kmeans.predict(embedding_tensor)[0]
        if cluster_label == 0:  # Ajustez ce numéro de cluster
            start_time, end_time = segments[i]
            audio_segment, sr = librosa.load(audio_file, sr=16000, offset=start_time, duration=end_time - start_time)
            segments_professor.append(audio_segment)
    return segments_professor

# Fonction pour sauvegarder l'audio combiné
def save_combined_audio(segments_professor, output_path):
    if segments_professor:
        combined_audio = np.concatenate(segments_professor, axis=0)
        sf.write(output_path, combined_audio, 16000)
        print("prof audio have been successfully saved.")
    else:
        print("prof segments not found.")

def process_with_diariation(audio_file, auth_token):
    embedding_dir = "embeddings/"
    output_dir = "output/"
    tmp_dir = "tmp/"
    speaker_embedding_model, diarization_pipeline = load_models(auth_token)
    diarization = perform_diarization(diarization_pipeline, audio_file)
    num_speakers = count_speakers(diarization)
    embeddings, segments = extract_embeddings(diarization, audio_file, speaker_embedding_model)
    kmeans = apply_kmeans(embeddings, num_speakers)
    saved_embeddings = [os.path.join(embedding_dir, f) for f in os.listdir(embedding_dir) if f.endswith(".pt")]
    print("Embeddings found: ", saved_embeddings)
    i = 0
    if (len(saved_embeddings) == 0):
        print("Error: No embeddings found")
        print("Please run the following command to generate embeddings: python add_embedding.py")
        return
    for embedding in saved_embeddings:
        professor_embedding = torch.load(embedding)
        if hasattr(professor_embedding, 'data'):
            professor_embedding_tensor = torch.tensor(professor_embedding.data).mean(dim=0)
        else:
            professor_embedding_tensor = professor_embedding
        segments_professor = identify_professor_segments(
            embeddings, segments, kmeans, audio_file, professor_embedding_tensor
        )
        combined_output = tmp_dir + os.path.basename(sys.argv[1]) + "_combined" + str(i) + ".wav"
        save_combined_audio(segments_professor, combined_output)
        original, translated = transcribe_with_external_translation(combined_output)
        tmp = os.path.basename(combined_output)
        f = open(output_dir + tmp + "translated.txt", "w")
        f.write(translated)
        f.close()
        i = i + 1
    return

def process_without_diariation(audio_file):
    output_dir = "output/"
    original, translated = transcribe_with_external_translation(audio_file)
    tmp = os.path.basename(audio_file)
    f = open(output_dir + tmp + "translated.txt", "w")
    f.write(translated)
    f.close()
    return

def main():
    if (len(sys.argv) < 2):
        print("program needs an audio argument")
        return
    print("Processing audio file: " + sys.argv[1])
    make_dirs()
    embedding_dir = "embeddings/"
    tmp_dir = "tmp/"
    output_dir = "output/"
    audio_file = denoise(sys.argv[1], tmp_dir + os.path.basename(sys.argv[1]) + "_denoised.wav")
    if (enable_diarization):
        process_with_diariation(audio_file)
    else:
        process_without_diariation(audio_file)
    return

if __name__ == "__main__":
    main()
