import numpy as np
import matplotlib.pyplot as plt
import os
import io
import base64
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model

chexnet_labels = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices"
]

def load_chexnet_model():
    model = DenseNet121(
        include_top=True,
        weights=None,
        input_shape=(224, 224, 3),
        classes=14
    )
    model.load_weights('brucechou1983_CheXNet_Keras_0.3.0_weights.h5')
    return model

chexnet_model = load_chexnet_model()

def predict_chexnet(img_path, age, sex, view, projection, target_size=(224, 224)):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    predictions = chexnet_model.predict(img_array)[0]
    print("Raw predictions:", predictions)

    # Prepare results
    results = []
    for i, label in enumerate(chexnet_labels):
        results.append({
            'ailment': label,
            'confidence': float(predictions[i]),
            'description': ''  # Optionally add descriptions
        })

    # Visualization (as base64) - Table only
    fig, ax = plt.subplots(figsize=(6, 5))
    data = []
    for i, label in enumerate(chexnet_labels):
        pred_prob = predictions[i]
        data.append([label, f"{pred_prob:.3f}"])
    col_labels = ["Pathology", "Prediction"]
    table = ax.table(cellText=data, colLabels=col_labels, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    ax.axis('off')
    plt.title(f"Patient Info - Age: {age}, Sex: {sex}, View: {view}, Projection: {projection}", fontsize=12)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    vis_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        'results': results,
        'visualization': vis_base64
    }
