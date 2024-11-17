import ollama
import pypdf
import pandas
import os

from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

UPLOAD_FOLDER      = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'xlsx', 'mp3'}

def allowed_path(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_pdf(path) -> str:
    reader = pypdf.PdfReader(path)
    output = ""

    for page in reader.pages:
        output += page.extract_text(
            extraction_mode="layout",
            layout_mode_scale_weight=1.2
        )

    return output

def extract_excel(path) -> str:
    return pandas.read_excel(path, index_col=None).to_csv(encoding="utf-8")

def extract_speech(path) -> str:
    return pipeline("automatic-speech-recognition", "openai/whisper-base")(path)['text']

def extract_text(path) -> str:
    with open(path, 'r') as file:
        return file.read()

handlers = {
    '.pdf'  : extract_pdf,
    '.txt'  : extract_text,
    '.mp3'  : extract_speech,
    '.xlsx' : extract_excel
}

@app.route("/api/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    name = file.filename

    if name == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_path(name):
        file.save(
            os.path.join(UPLOAD_FOLDER, name)
        )
        
        return jsonify({"message": "File uploaded successfully", "name": name}), 200
    else:
        return jsonify({"error": "Invalid file type"}), 400
    
@app.route("/api/summarize", methods=["POST"])
def summarize():
    path = UPLOAD_FOLDER + "/" + request.form['filename']

    if not os.path.exists(path):
        return jsonify({"error": "File does not exist - did you upload it first?"}), 400

    text = handlers[os.path.splitext(path)[1]](path)

    output = ollama.chat(
        model='llama3.1',
        options={'temperature': 0},
        messages=[
            {'role': 'system', 'content': "You are a document summary generator. The user will present you with a document; you should generate a summary of the document without any framing (e.g. 'here is your summary'.)"},
            {'role': 'user', 'content': "Summarize the following document.\n===\n%s\n===" % (text)}
        ],
    )

    return jsonify({"summary": output['message']['content']}), 200

@app.route("/api/interrogate", methods=["POST"])
def interrogate():
    path  = UPLOAD_FOLDER + "/" + request.form['filename']
    query = request.form['query']

    if not os.path.exists(path):
        return jsonify({"error": "File does not exist - did you upload it first?"}), 400
    
    text = handlers[os.path.splitext(path)[1]](path)

    output = ollama.chat(
        model='llama3.1',
        options={'temperature': 0},
        messages=[
            {'role': 'system', 'content': "You are a document querying system. The user will present you with a document and a question about the document; you should answer their question, citing portions of the original text as needed. Avoid framing, such as 'according to the document.'"},
            {'role': 'user', 'content': "Document:\n===\n%s\n===\nQuestion: %s" % (text, query)}
        ],
    )

    return jsonify({"answer": output['message']['content']}), 200

if __name__ == "__main__":
    app.run(debug=True)