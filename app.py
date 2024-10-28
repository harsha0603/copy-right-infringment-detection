from flask import Flask, request, render_template, redirect, url_for, send_file, flash
import os
from datetime import datetime
from copyright_infringement.components.similarity_detection import SimilarityDetection
from copyright_infringement.utils.main_utils import extract_text_from_pdf, save_pdf
from copyright_infringement.entity.config_entity import SimilarityDetectionConfig, EmbeddingConfig

app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['REPORT_FOLDER'] = "reports"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

# Initialize configuration for similarity detection and embeddings
similarity_config = SimilarityDetectionConfig(similarity_threshold=0.75)  # Customize as needed
embedding_config = EmbeddingConfig(model_name="bert-base-uncased")  # Use your desired model

# Initialize similarity detection component
similarity_detector = SimilarityDetection(config=similarity_config, embedding_config=embedding_config)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        flash("No file part in the request")
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash("No selected file")
        return redirect(url_for('index'))
    
    if file and file.filename.lower().endswith('.pdf'):
        pdf_path = save_pdf(file, app.config['UPLOAD_FOLDER'])
        
        # Extract text from the uploaded PDF
        extracted_text = extract_text_from_pdf(pdf_path)
        
        # Run similarity detection
        similarity_result = similarity_detector.check_similarity(extracted_text)
        
        # Generate and save report
        report_path = generate_report(similarity_result)
        
        # Update this line to extract the report filename
        return render_template("report.html", result=similarity_result, report_filename=os.path.basename(report_path))
    else:
        flash("Please upload a valid PDF file")
        return redirect(url_for('index'))

def generate_report(similarity_result):
    # Define report path
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    report_filename = f"report_{timestamp}.txt"
    report_path = os.path.join(app.config['REPORT_FOLDER'], report_filename)
    
    # Create report content
    with open(report_path, 'w') as f:
        if similarity_result.is_copyright_infringement:
            f.write("Potential Copyright Infringement Detected!\n")
            f.write(f"Similarity Score: {similarity_result.similarity_score}\n")
            f.write(f"Similar Text: {similarity_result.similar_text}\n")
        else:
            f.write("No Infringement Detected.\n")
            f.write(f"Similarity Score: {similarity_result.similarity_score}\n")
    
    print(f"Report generated: {report_path}")  # Debug statement
    return report_path

@app.route('/download_report/<path:filename>', methods=['GET'])
def download_report(filename):
    file_path = os.path.join(app.config['REPORT_FOLDER'], filename)
    if not os.path.isfile(file_path):
        flash("Report not found!")
        return redirect(url_for('index'))
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
