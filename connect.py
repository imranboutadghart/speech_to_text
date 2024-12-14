#!/home/iboutadg/tests/se_project/se/bin/python3
from flask import Flask, request
import subprocess
import Timer
timer = Timer.Timer()

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if (not (file.filename.endswith(".wav") or file.filename.endswith(".mp3"))):
        return 'Invalid file type', 400
    file.save(file.filename)
    script_name = "no_script.py"
    if 'script' in request.form and request.form['script'] in ['add_embedding', 'process']:
        script_name = request.form['script'] + '.py'
    print("Running script: " + script_name + " with file: " + file.filename, flush=True)
    result = subprocess.run(['python3', script_name, file.filename], capture_output=True, text=True)
    try:
        with open("errlog.txt", "w") as f:
            f.write(result.stderr)
    except:
        pass
    return result.stdout

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
