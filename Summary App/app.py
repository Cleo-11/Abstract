import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from flask import Flask, request, jsonify, send_from_directory
import pdfplumber
from docx import Document
import re
import requests
import nltk
import logging
from typing import Dict, Any
from functools import lru_cache
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
AIRTABLE_API_KEY = 'patLJEdqVH5X9cD1b.6d9149ffb0adae3667ff3cde09b64b11d668d27f73b3448770ae792b99c62791'
BASE_ID = 'apptD8crqpruahv2R'
TABLE_NAME = 'Summary Deets'

# Word length options
WORD_LENGTH_OPTIONS: Dict[str, int] = {
    'small': 100,
    'medium': 200,
    'large': 300
}

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True

@lru_cache(maxsize=None)
def get_airtable_headers() -> Dict[str, str]:
    return {
        'Authorization': f'Bearer {AIRTABLE_API_KEY}',
        'Content-Type': 'application/json'
    }

def save_summary_to_airtable(file_name: str, summary: str) -> Dict[str, Any]:
    url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
    data = {
        'fields': {
            'File Name': file_name,
            'Summary': summary
        }
    }
    try:
        response = requests.post(url, headers=get_airtable_headers(), json=data)
        response.raise_for_status()
        logging.info(f"Successfully saved to Airtable: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error saving to Airtable: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response content: {e.response.content}")
        logging.error(f"Request data: {data}")
        return {}

def extract_text_from_pdf(pdf_file) -> str:
    with pdfplumber.open(pdf_file) as pdf:
        return '\n'.join(page.extract_text() or '' for page in pdf.pages)

def extract_text_from_docx(docx_file) -> str:
    doc = Document(docx_file)
    return '\n'.join(para.text.strip() for para in doc.paragraphs if para.text.strip())

def preprocess_text(text: str) -> str:
    # Remove URLs
    text = re.sub(r'(https?://\S+|www\.\S+)', '', text)
    
    # Remove specific patterns
    patterns_to_remove = [
        r'\b(?:doi|arxiv|fig|table|hal|ids|isbn|issn|pp)\b.*?(?=\s|\n|$)',
        r'\b(?:submitted|accepted|published|conference|journal)\b.*?(?=\s|\n|$)',
        r'\[[0-9]+\]',
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove non-printable characters and extra whitespace
    text = ''.join(char for char in text if char.isprintable())
    return re.sub(r'\s+', ' ', text).strip()

def postprocess_summary(summary: str) -> str:
    # Remove non-printable characters
    summary = ''.join(char for char in summary if char.isprintable())
    
    # Replace multiple spaces with a single space
    summary = re.sub(r'\s+', ' ', summary)
    
    # Add spaces after periods if missing
    summary = re.sub(r'\.(\S)', '. \1', summary)
    
    # Split incorrectly joined words
    summary = re.sub(r'([a-z])([A-Z])', r'\1 \2', summary)
    
    # Fix common OCR errors and split long words
    words = summary.split()
    processed_words = []
    for word in words:
        if len(word) > 15:  # Split very long words
            processed_words.extend(re.findall(r'.{1,15}', word))
        else:
            # Fix common OCR errors
            word = word.replace('ofthe', 'of the')
            word = word.replace('andthe', 'and the')
            word = word.replace('inthe', 'in the')
            word = word.replace('tothe', 'to the')
            processed_words.append(word)
    
    summary = ' '.join(processed_words)
    
    # Capitalize the first letter of each sentence
    summary = '. '.join(s.capitalize() for s in summary.split('. '))
    
    # Remove spaces before punctuation
    summary = re.sub(r'\s([,.!?;:](?:\s|$))', r'\1', summary)
    
    # Ensure proper spacing around parentheses
    summary = re.sub(r'\s*\(\s*', ' (', summary)
    summary = re.sub(r'\s*\)\s*', ') ', summary)
    
    # Remove any remaining non-alphanumeric characters (except punctuation)
    summary = re.sub(r'[^\w\s.,!?;:()-]', '', summary)
    
    return summary.strip()

def generate_summary(text: str, word_length: int) -> str:
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=word_length * 2,  # Increase max_length to give more room for postprocessing
            min_length=word_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = postprocess_summary(summary)
    
    # Ensure the summary doesn't exceed the word limit
    words = summary.split()
    return ' '.join(words[:word_length])

@app.route('/summarize', methods=['POST'])
def summarize_paper():
    file = request.files['file']
    word_length_option = request.form.get('word_length', 'medium')
    
    logging.info(f"Received file: {file.filename}, word length option: {word_length_option}")
    
    word_length = WORD_LENGTH_OPTIONS.get(word_length_option, 200)

    try:
        if file.filename.endswith('.pdf'):
            paper_text = extract_text_from_pdf(file)
        elif file.filename.endswith('.docx'):
            paper_text = extract_text_from_docx(file)
        else:
            return jsonify({"error": "Unsupported file format. Please provide a PDF or DOCX file."}), 400

        clean_text = preprocess_text(paper_text)
        summary = generate_summary(clean_text, word_length)

        airtable_response = save_summary_to_airtable(file.filename, summary)
        if not airtable_response:
            logging.warning("Failed to save summary to Airtable")

        return jsonify({
            "summary": summary, 
            "file_name": file.filename, 
            "requested_length": word_length_option
        })
    except Exception as e:
        logging.error(f"Error in summarize_paper: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/past_summaries', methods=['GET'])
def past_summaries():
    url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
    try:
        response = requests.get(url, headers=get_airtable_headers())
        response.raise_for_status()
        data = response.json()
        logging.info(f"Successfully fetched past summaries from Airtable. Data: {data}")
        return jsonify(data)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching past summaries: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response content: {e.response.content}")
        return jsonify({"error": "Failed to fetch past summaries"}), 500

@app.route('/', methods=['GET'])
def serve_homepage():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>', methods=['GET'])
def serve_static_file(path):
    return send_from_directory('static', path)

@app.route('/delete_summary/<string:record_id>', methods=['DELETE'])
def delete_summary(record_id):
    url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}/{record_id}"
    try:
        response = requests.delete(url, headers=get_airtable_headers())
        response.raise_for_status()
        return jsonify({"success": True, "message": "Summary deleted successfully"}), 200
    except requests.exceptions.RequestException as e:
        logging.error(f"Error deleting summary: {e}")
        return jsonify({"error": "Failed to delete summary"}), 500

if __name__ == '__main__':
    app.run(debug=True)