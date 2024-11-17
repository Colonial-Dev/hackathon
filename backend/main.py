import os
import tempfile

import ollama
import pypdf
import pandas

from flask import Flask, request, jsonify
from transformers import pipeline
from uuid import uuid4

app = Flask(__name__)

UPLOAD_FOLDER      = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'xlsx', 'mp3'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        uuid = uuid4()
        path = os.path.join(UPLOAD_FOLDER, f"{uuid}")
        text = ""

        with tempfile.NamedTemporaryFile() as buffer, open(path, "w") as processed:
            file.save(buffer)
            text = handlers[os.path.splitext(name)[1]](buffer.name)
            processed.write(text)
                
        return jsonify(
            {
                "message": "File uploaded successfully",
                "file_id": str(uuid),
                "text_preview": text[:200],
                "text_length": len(text)
            }
        ), 200
    else:
        return jsonify({"error": "Invalid file type"}), 400
    
@app.route("/api/summarize/<file_id>")
def summarize(file_id):
    path = os.path.join(UPLOAD_FOLDER, file_id)

    if not os.path.exists(path):
        return jsonify({"error": "File does not exist - did you upload it first?"}), 400

    text = open(path, "r").read()

    output = ollama.chat(
        model='llama3.1',
        options={'temperature': 0},
        messages=[
            {'role': 'system', 'content': "You are a document summary generator. The user will present you with a document; you should generate a summary of the document without any framing (e.g. 'here is your summary'.)"},
            {'role': 'user', 'content': "Summarize the following document.\n===\n%s\n===" % (text)}
        ],
    )

    return jsonify({"summary": output['message']['content']}), 200

@app.route("/api/interrogate/<file_id>", methods=["POST"])
def interrogate(file_id):
    path = os.path.join(UPLOAD_FOLDER, file_id)

    if not os.path.exists(path):
        return jsonify({"error": "File does not exist - did you upload it first?"}), 400
    
    if request.json is None:
        return jsonify({"error": "Expected JSON request body"}), 400
    
    query = request.json.get('question', '')
    text  = open(path, "r").read()

    output = ollama.chat(
        model='llama3.1',
        options={'temperature': 0},
        messages=[
            {'role': 'system', 'content': "You are a document querying system. The user will present you with a document and a question about the document; you should answer their question, citing portions of the original text as needed. Avoid framing, such as 'according to the document.'"},
            {'role': 'user', 'content': "Document:\n===\n%s\n===\nQuestion: %s" % (text, query)}
        ],
    )

    return jsonify({"answer": output['message']['content']}), 200

@app.route("/api/sentiment/<file_id>")
def sentiment(file_id):
    analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def split_text(text, max_length=512):
        sentences = text.split('.')
        chunks, current_chunk = [], []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(analyzer.tokenizer.tokenize(sentence))
            if current_length + sentence_length > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def aggregate_sentiments(sentiment_output):
        label_counts = {"POSITIVE": 0, "NEGATIVE": 0}
        score_sums = {"POSITIVE": 0.0, "NEGATIVE": 0.0}
        
        for chunk_result in sentiment_output:
            for result in chunk_result:
                label = result['label']
                score = result['score']
                label_counts[label] += 1
                score_sums[label] += score
        
        avg_scores = {label: (score_sums[label] / label_counts[label] if label_counts[label] > 0 else 0)
                    for label in label_counts}
        
        overall_label = max(avg_scores, key=avg_scores.get)

        return {
            "overall_sentiment": overall_label,
            "average_scores": avg_scores,
            "chunk_counts": label_counts
        }
    
    path = os.path.join(UPLOAD_FOLDER, file_id)

    if not os.path.exists(path):
        return jsonify({"error": "File does not exist - did you upload it first?"}), 400
    
    text   = open(path, "r").read()
    chunks = split_text(text)
    result = [analyzer(chunk) for chunk in chunks]
    summed = aggregate_sentiments(result)

    return jsonify(summed), 200


if __name__ == "__main__":
    app.run(debug=True)