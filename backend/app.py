from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
from flask_cors import CORS
import PyPDF2  # Add this import

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            text_content = []
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())
            
            return '\n'.join(text_content)
    except Exception as e:
        print(f"Error extracting PDF text: {str(e)}")
        return ""

@app.route('/')
def home():
    return jsonify({"message": "API is working!"})

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload endpoint hit!")  # Debug print
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Generate unique ID and save file
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        file.save(file_path)
        
        print(f"File saved: {file_path}")  # Debug print
        
        # Extract text from PDF
        if filename.lower().endswith('.pdf'):
            text_content = extract_text_from_pdf(file_path)
            text_preview = text_content[:200]  # First 200 characters as preview
            
            response_data = {
                'file_id': file_id,
                'message': 'File uploaded successfully',
                'text_preview': text_preview,
                'text_length': len(text_content)
            }
        else:
            response_data = {
                'file_id': file_id,
                'message': 'File uploaded successfully',
                'warning': 'Not a PDF file - no text extracted'
            }
        
        print("Response data:", response_data)  # Debug print
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error in upload: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500

@app.route('/status/<file_id>', methods=['GET'])
def get_status(file_id):
    # Mock status for testing
    return jsonify({
        'status': 'processing',
        'progress': 50
    })

@app.route('/summary/<file_id>', methods=['GET'])
def get_summary(file_id):
    # Mock summary for testing
    return jsonify({
        'filename': 'Test Document',
        'fileType': 'PDF',
        'pages': '5',
        'summary': 'This is a test summary of the document.'
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '')
    file_id = data.get('fileId', '')
    
    # Mock answer for testing
    return jsonify({
        'answer': f'This is a test answer to your question: {question}'
    })

if __name__ == '__main__':
    print("Starting Flask server on port 5000...")  # Debug print
    app.run(debug=True, port=5000)