from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import time
import uuid

# Import your three agents
from preprocessing_agent import preprocess_dat_file
from classification_agent import classify_heartbeats
from communication_agent import (
    analyze_predictions_expert, 
    generate_doctor_report_expert, 
    generate_patient_report_expert_clean
)

app = Flask(__name__)

# Configure a temporary folder for uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'dat'}

def allowed_file(filename):
    """Checks if the uploaded file has the allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    """Renders the home page."""
    return render_template('home.html')

@app.route('/process', methods=['POST'])
def process():
    """Handles the file upload and the analysis pipeline."""
    if 'ecg_file' not in request.files:
        return redirect(url_for('home'))
        
    file = request.files['ecg_file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('home'))

    unique_filename_no_ext = str(uuid.uuid4())
    record_path_no_ext = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename_no_ext)
    
    file_path_dat = record_path_no_ext + '.dat'
    file.save(file_path_dat)
    
    try:
        file_size_bytes = os.path.getsize(file_path_dat)
        num_samples = file_size_bytes // 3 
    except OSError:
        return "Error reading uploaded file size.", 500

    create_dynamic_header(record_path_no_ext, unique_filename_no_ext, num_samples)

    preprocessing_output = preprocess_dat_file(record_path_no_ext)
    
    os.remove(record_path_no_ext + '.dat')
    os.remove(record_path_no_ext + '.hea')

    if not preprocessing_output:
        return "Error during preprocessing. The uploaded file may be corrupt or in an unexpected format.", 400

    predicted_labels = classify_heartbeats(preprocessing_output['signals'])
    
    if predicted_labels is None:
        return "Error during classification.", 400

    expert_analysis = analyze_predictions_expert(
        predictions=predicted_labels,
        beat_locations=preprocessing_output['r_peaks'],
        fs=preprocessing_output['fs']
    )
    
    doctor_report = generate_doctor_report_expert(expert_analysis)
    patient_report = generate_patient_report_expert_clean(expert_analysis)

    if doctor_report is None or patient_report is None:
        return "Analysis failed. Could not generate reports. The ECG signal may be too short or too noisy.", 400

    return render_template('processing.html', doctor_report_data=doctor_report, patient_report_data=patient_report)

def create_dynamic_header(record_path, record_name, num_samples):
    """Creates a basic .hea file that mimics the MIT-BIH format."""
    sampling_frequency = 360
    header_content = f"{record_name} 2 {sampling_frequency} {num_samples}\n"
    header_content += f"{record_name}.dat 212 200 11 1024 0 0 0 MLII\n"
    header_content += f"{record_name}.dat 212 200 11 1024 0 0 0 V1\n"
    with open(record_path + '.hea', 'w') as f:
        f.write(header_content)

if __name__ == '__main__':
    app.run(debug=True)