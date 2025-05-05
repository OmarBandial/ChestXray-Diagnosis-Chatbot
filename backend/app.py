
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
import time

# In a real application, these would be actual ML modules
# import load_model
# import LLM

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Mock database for sessions (in production, use a real database)
sessions = {}

# Routes
@app.route('/api/upload-xray', methods=['POST'])
def upload_xray():
    """Handle X-ray image upload"""
    if 'xray' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['xray']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Create a unique session ID
        session_id = str(uuid.uuid4())
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(file_path)
        
        # Store session info
        sessions[session_id] = {
            'xray_path': file_path,
            'created_at': time.time(),
            'patient_data': None,
            'diagnosis_results': None
        }
        
        return jsonify({
            'success': True, 
            'sessionId': session_id,
            'message': 'X-ray uploaded successfully'
        })
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/api/patient-data', methods=['POST'])
def submit_patient_data():
    """Handle patient data submission"""
    data = request.json
    session_id = data.get('sessionId')
    
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid or expired session'}), 400
    
    # Store patient data in session
    sessions[session_id]['patient_data'] = {
        'age': data.get('age'),
        'gender': data.get('gender'),
        'symptoms': data.get('symptoms', []),
        'painLevel': data.get('painLevel'),
        'medicalHistory': data.get('medicalHistory', ''),
        'medications': data.get('medications', [])
    }
    
    # In a real app, we'd call the model here:
    # results = load_model.predict(
    #     xray_path=sessions[session_id]['xray_path'], 
    #     patient_data=sessions[session_id]['patient_data']
    # )
    
    # Mock diagnosis results for demonstration
    mock_results = [
        {
            'ailment': 'Pneumonia',
            'confidence': 0.87,
            'description': 'Inflammation of the lungs caused by infection.'
        },
        {
            'ailment': 'Pleural Effusion',
            'confidence': 0.45,
            'description': 'Buildup of fluid between the lungs and chest cavity.'
        },
        {
            'ailment': 'Atelectasis',
            'confidence': 0.32,
            'description': 'Collapse or closure of a lung resulting in reduced or absent gas exchange.'
        }
    ]
    
    # Store diagnosis results
    sessions[session_id]['diagnosis_results'] = mock_results
    
    return jsonify({
        'success': True,
        'results': mock_results
    })

@app.route('/api/diagnosis/<session_id>', methods=['GET'])
def get_diagnosis(session_id):
    """Retrieve diagnosis results for a session"""
    if session_id not in sessions:
        return jsonify({'error': 'Invalid or expired session'}), 400
    
    if not sessions[session_id].get('diagnosis_results'):
        return jsonify({'error': 'Diagnosis not yet performed'}), 404
    
    # Return the diagnosis results
    return jsonify({
        'success': True,
        'results': sessions[session_id]['diagnosis_results']
    })

@app.route('/api/chat/<session_id>', methods=['POST'])
def chat_with_ai(session_id):
    """Handle AI chat interaction"""
    if session_id not in sessions:
        return jsonify({'error': 'Invalid or expired session'}), 400
    
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    
    # In a real app, this would use the LLM module:
    # ai_response = LLM.generate_response(
    #     user_message=user_message,
    #     diagnosis_results=sessions[session_id]['diagnosis_results'],
    #     patient_data=sessions[session_id]['patient_data']
    # )
    
    # Mock AI responses for demonstration
    ai_responses = [
        "Based on your X-ray, I can see signs of inflammation in the lower right lung. This is consistent with early-stage pneumonia, which aligns with the symptoms you've reported.",
        "Your symptoms of shortness of breath and chest pain are common with pleural effusion. The fluid buildup visible on your X-ray explains these symptoms.",
        "While your symptoms are concerning, the X-ray doesn't show any severe abnormalities. This could be bronchitis rather than pneumonia, which wouldn't necessarily appear on an X-ray.",
        "The patterns visible in your lung fields suggest you may have had this condition for some time. Have you experienced these symptoms before?",
        "Your medical history of asthma is important to consider here, as it can sometimes cause similar symptoms. However, the X-ray findings suggest this is a separate issue requiring different treatment."
    ]
    
    import random
    ai_response = random.choice(ai_responses)
    
    return jsonify({
        'success': True,
        'message': ai_response,
        'timestamp': time.time()
    })

# Cleanup old sessions periodically (in production, this would be a scheduled task)
@app.before_request
def cleanup_old_sessions():
    """Remove sessions older than 1 hour"""
    current_time = time.time()
    expired_sessions = []
    
    for session_id, session_data in sessions.items():
        if current_time - session_data['created_at'] > 3600:  # 1 hour
            expired_sessions.append(session_id)
            
            # Remove the uploaded file
            if os.path.exists(session_data['xray_path']):
                os.remove(session_data['xray_path'])
    
    for session_id in expired_sessions:
        sessions.pop(session_id, None)

# Serve uploaded files (for development only)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
