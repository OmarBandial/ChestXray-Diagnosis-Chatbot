
import requests
import base64
import json

# Load and encode image as base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Query Ollama with or without image
def query_ollama(prompt, image_path=None):
    payload = {
        "model": "gemma3:4b",
        "prompt": prompt
    }

    if image_path:
        image_base64 = encode_image_to_base64(image_path)
        payload["images"] = [image_base64]

    response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)

    if response.status_code == 200:
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_data = json.loads(line.decode("utf-8"))
                    full_response += json_data.get("response", "")
                except json.JSONDecodeError as e:
                    print("Decode error:", e)
        return full_response
    else:
        print("Error:", response.status_code, response.text)
        return None

def generate_response(user_message, diagnosis_results, patient_data):
    """
    Generate a contextual response based on the user's message and the diagnosis context.
    
    Args:
        user_message (str): The user's message/question
        diagnosis_results (list): The diagnosis results from the model
        patient_data (dict): The patient information
        
    Returns:
        str: AI-generated response
    """
    # Format the context information into a detailed prompt for the LLM
    prompt = f"""
You are a medical assistant AI. Below are X-ray analysis results and patient information.
Please provide a helpful, accurate, and compassionate response to the user's question.

PATIENT INFORMATION:
- Age: {patient_data.get('age', 'Unknown')}
- Gender: {patient_data.get('gender', 'Unknown')}
- Pain Level: {patient_data.get('painLevel', 'Unknown')}/10
- Symptoms: {', '.join(patient_data.get('symptoms', ['None reported']))}
- Medical History: {patient_data.get('medicalHistory', 'None provided')}
- Medications: {', '.join(patient_data.get('medications', ['None reported']))}

X-RAY ANALYSIS RESULTS:
"""

    # Add each condition with its probability
    for result in diagnosis_results:
        prompt += f"- {result['ailment']}: {result['confidence']:.2f} probability\n"
    
    prompt += f"\nUSER QUESTION: {user_message}\n\nPlease provide a detailed, helpful response that addresses the user's question specifically in relation to their X-ray results and medical information. Include relevant medical information while being compassionate and clear."
    
    # Query the LLM
    return query_ollama(prompt)
