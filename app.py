from flask import Flask, request, render_template, redirect, url_for, send_file, flash
import os
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from copyright_infringement.components.similarity_detection import SimilarityDetection
from copyright_infringement.utils.main_utils import extract_text_from_pdf, save_pdf
from copyright_infringement.entity.config_entity import SimilarityDetectionConfig, EmbeddingConfig

# Flask App Initialization
app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['REPORT_FOLDER'] = "reports"
app.config['IMAGE_UPLOAD_FOLDER'] = "static/image_uploads"  # Corrected image upload folder path
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_UPLOAD_FOLDER'], exist_ok=True)

# Paths for Image Similarity
RAW_IMAGES_PATH = "./artifacts/raw_images"
EMBEDDINGS_PATH = "./artifacts/embeddings"

# Text Similarity Config
similarity_config = SimilarityDetectionConfig(similarity_threshold=0.75)  # Customize as needed
embedding_config = EmbeddingConfig(model_name="bert-base-uncased")  # Use your desired model
similarity_detector = SimilarityDetection(config=similarity_config, embedding_config=embedding_config)

# Image Similarity Config
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")  # Pretrained model for embeddings

# -----------------------------------------------
# Helper Functions for Image Similarity Detection
# -----------------------------------------------
def preprocess_image(image_path):
    """Preprocesses an image for ResNet50."""
    image = load_img(image_path, target_size=(224, 224))  # Load and resize image
    img_array = img_to_array(image)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for ResNet50
    return img_array


def get_embedding(image_path):
    """Generate embedding for a single image."""
    image_array = preprocess_image(image_path)
    embedding = base_model.predict(image_array)
    return embedding


def check_image_infringement(input_image_path, threshold=0.85):
    """
    Check if the input image matches any stored embeddings and return the image with the highest similarity.
    """
    input_embedding = get_embedding(input_image_path)

    # Initialize variables to store maximum similarity and corresponding file
    max_similarity = 0
    most_similar_file = None

    # Load all stored embeddings and calculate similarity
    for embedding_file in os.listdir(EMBEDDINGS_PATH):
        embedding_path = os.path.join(EMBEDDINGS_PATH, embedding_file)

        # Check if the file is an embedding file
        if embedding_file.endswith("_embeddings.npy"):
            # Extract the base name of the image from the embedding filename
            image_name = embedding_file.replace("_embeddings.npy", "")

            # Load stored embedding
            stored_embedding = np.load(embedding_path)

            # Calculate cosine similarity
            similarity = cosine_similarity(input_embedding.reshape(1, -1), stored_embedding.reshape(1, -1))[0][0]

            # Update max similarity if a higher similarity is found
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_file = (image_name, similarity)

    # Check if the highest similarity exceeds the threshold
    if max_similarity >= threshold:
        return most_similar_file, max_similarity
    else:
        return None, 0



# -----------------------------------
# Routes for Text Similarity Detection
# -----------------------------------
@app.route('/')
def index():
    return render_template("index.html",datetime=datetime.now())


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        flash("No file part in the request.")
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash("No file selected.")
        return redirect(url_for('index'))
    
    # Get the file extension
    file_extension = os.path.splitext(file.filename)[-1].lower()

    if file_extension == '.pdf':
        # Save the PDF and process it
        pdf_path = save_pdf(file, app.config['UPLOAD_FOLDER'])
        extracted_text = extract_text_from_pdf(pdf_path)
        similarity_result = similarity_detector.check_similarity(extracted_text)
        
        # Generate and return the PDF report
        report_path = generate_report(similarity_result)
        return render_template("report.html", result=similarity_result, report_filename=os.path.basename(report_path),datetime=datetime.now())
    
    elif file_extension in ['.png', '.jpg', '.jpeg']:
        # Save the image and process it
        unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        image_path = os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], unique_filename)
        file.save(image_path)
        
        most_similar_files, max_similarity = check_image_infringement(image_path, threshold=0.85)
        
        if most_similar_files:
            return render_template(
                "image_report.html",
                files=most_similar_files,
                similarity=max_similarity,
                uploaded_image=os.path.basename(image_path)
            )
        else:
            flash("No infringement detected. The image is unique.")
            return redirect(url_for('index'))
    
    else:
        flash("Unsupported file type. Please upload a PDF or image file.")
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
            f.write(f"Sourced From: {similarity_result.similar_text}\n")
        else:
            f.write("No Infringement Detected.\n")
            f.write(f"Similarity Score: {similarity_result.similarity_score}\n")
    
    return report_path


@app.route('/download_report/<path:filename>', methods=['GET'])
def download_report(filename):
    file_path = os.path.join(app.config['REPORT_FOLDER'], filename)
    if not os.path.isfile(file_path):
        flash("Report not found!")
        return redirect(url_for('index'))
    return send_file(file_path, as_attachment=True)


# -----------------------------------
# Routes for Image Similarity Detection
# -----------------------------------
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash("No file part in the request")
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash("No selected file")
        return redirect(url_for('index'))
    
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Save file to the defined image upload folder
        image_path = os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], file.filename)  # Corrected path
        file.save(image_path)
        
        # Run similarity detection
        most_similar_file, max_similarity = check_image_infringement(image_path, threshold=0.85)
        
        if most_similar_file:
            flash(f"Potential infringement detected! Similarity: {max_similarity}")
            # Pass the filename to the template for rendering
            return render_template('image_report.html', 
                                   uploaded_image=file.filename,  # Make sure this is passed
                                   most_similar_file=most_similar_file, 
                                   similarity=max_similarity,datetime=datetime.now())
        else:
            flash("No infringement detected. Image is unique.")
            return redirect(url_for('index'))
    else:
        flash("Please upload a valid image file (PNG, JPG, JPEG)")
        return redirect(url_for('index'))


# -----------------------------------
# Main Entry Point
# -----------------------------------
if __name__ == "__main__":
    app.run(debug=True)
