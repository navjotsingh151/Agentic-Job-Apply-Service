import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from agents.cover_letter_agent import run_cover_letter_agent
import tempfile
import textract  # For extracting text from PDFs/DOCs

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_resume_text(file_path):
    try:
        return textract.process(file_path).decode('utf-8')
    except Exception as e:
        return f"Error extracting resume text: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        user_prompt = request.form.get('user_prompt')
        job_description = request.form.get('job_description')
        resume_text = ""

        # Process uploaded resume
        uploaded_file = request.files.get('resume_file')
        if uploaded_file and allowed_file(uploaded_file.filename):
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)
            resume_text = extract_resume_text(file_path)

        # Combine everything for agent input
        full_prompt = f"""
User Prompt: {user_prompt}

Job Description: {job_description}

Resume Content: {resume_text}
"""

        result = run_cover_letter_agent(full_prompt)

    return render_template('index.html', result=result)

if __name__=="__main__":
    app.run(debug=True)
