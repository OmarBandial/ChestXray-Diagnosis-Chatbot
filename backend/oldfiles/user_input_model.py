import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
import tensorflow as tf

# Load CheXNet model with weights
def load_chexnet_model():
    model = DenseNet121(
        include_top=True,
        weights=None,
        input_shape=(224, 224, 3),
        classes=14
    )
    model.load_weights(r'backend\brucechou1983_CheXNet_Keras_0.3.0_weights.h5')
    return model

# CheXNet label list
chexnet_labels = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices"
]

# Predict and display result for a single image
def predict_and_display_single_image(model, img_path, age, sex, view, projection, target_size=(224, 224)):
    if not os.path.exists(img_path):
        print(f"Error: File not found at {img_path}")
        return

    # Load and preprocess image
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # Predict
    predictions = model.predict(img_array)[0]

    # Display image and prediction table
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title(f"Image: {os.path.basename(img_path)}", fontsize=12)
    axes[0].axis('off')

    data = []
    cell_colors = []

    for i, label in enumerate(chexnet_labels):
        pred_prob = predictions[i]
        actual_val = 0  # Default to 0 unless you have ground-truth labels
        data.append([label, f"{pred_prob:.3f}", actual_val])
        if (pred_prob > 0.5 and actual_val == 1) or (pred_prob <= 0.5 and actual_val == 0):
            cell_colors.append(['lightgreen'] * 3)
        else:
            cell_colors.append(['lightcoral'] * 3)

    col_labels = ["Pathology", "Predicted", "True"]
    table = axes[1].table(cellText=data, colLabels=col_labels, cellColours=cell_colors,
                          cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    axes[1].axis('off')

    # Additional info
    plt.suptitle(f"Patient Info - Age: {age}, Sex: {sex}, View: {view}, Projection: {projection}", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("---- CheXNet X-ray Prediction ----")
    img_path = input("Enter the path to the X-ray image: ")
    age = input("Enter patient's age: ")
    sex = input("Enter patient's sex (M/F): ")
    view = input("Enter image view (Frontal/Lateral): ")
    projection = input("Enter projection (AP/PA): ")

    model = load_chexnet_model()
    predict_and_display_single_image(model, img_path, age, sex, view, projection)
