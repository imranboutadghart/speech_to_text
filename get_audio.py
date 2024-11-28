#!/home/iboutadg/tests/se_project/se/bin/python3
# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import sys
from datetime import datetime
import subprocess

def save_audio():
    output_path = datetime.now().strftime("%Y%m%d%H%M%S") + ".wav"
    frequency = 16000
    duration = 5
    recording = sd.rec(int(duration * frequency), samplerate = frequency, channels = 2)
    sd.wait()
    write(output_path, frequency, recording)
    return output_path
if __name__ == "__main__":
    output_path = save_audio()
    subprocess.run(["./project.py", output_path])
