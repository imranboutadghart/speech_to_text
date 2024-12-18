#!/home/iboutadg/tests/se_project/se/bin/python3
from flask import Flask, request
from flask_cors import CORS
import subprocess
import os

UPLOAD_FOLDER = "uploaded/"  # Define the directory for uploads
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the directory exists
keep_uploaded = True

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if (not (file.filename.endswith(".wav") or file.filename.endswith(".mp3"))):
        return 'Invalid file type', 400
    filepath = UPLOAD_FOLDER + file.filename
    file.save(filepath)
    script_name = "no_script.py"
    if 'script' in request.form and request.form['script'] in ['add_embedding', 'process']:
        script_name = request.form['script'] + '.py'
    print("Running script: " + script_name + " with file: " + filepath, flush=True)
    result = subprocess.run(['python3', script_name, filepath], capture_output=True, text=True)
    try:
        with open("errlog.txt", "w") as f:
            f.write(result.stderr)
    except:
        pass
    if (not keep_uploaded):
        for filename in os.listdir(tmp_dir):
            file_path = os.path.join(tmp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except:
                pass
    return result.stdout

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
