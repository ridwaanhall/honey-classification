import os
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import JsonResponse
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Define paths to the model and scaler files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'base', 'model')

# Custom function to load the model without the original optimizer
def load_model_without_optimizer(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load the model and scaler
model = load_model_without_optimizer(os.path.join(MODEL_DIR, 'honey_classifier_model.h5'))
scaler = StandardScaler()
scaler.mean_ = np.load(os.path.join(MODEL_DIR, 'scaler.npy'))
scaler.scale_ = np.load(os.path.join(MODEL_DIR, 'scaler_scale.npy'))

def extract_rgb(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img_array = np.array(img)
        avg_color = img_array.mean(axis=(0, 1))
    return avg_color

def predict_honey_color(image_path):
    avg_color = extract_rgb(image_path)
    sample = np.array([avg_color])
    sample = scaler.transform(sample)
    prediction = model.predict(sample)[0][0]  # Get the raw output
    madu_akasia_prob = 1 - prediction
    madu_hutan_prob = prediction
    return madu_akasia_prob, madu_hutan_prob

def check(request):
    return render(request, 'check.html')

def predict(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        file_name = default_storage.save(file.name, ContentFile(file.read()))
        file_path = os.path.join(default_storage.location, file_name)

        # Make prediction
        madu_akasia_prob, madu_hutan_prob = predict_honey_color(file_path)

        # Clean up
        default_storage.delete(file_name)

        # return JsonResponse({
        #     'madu_akasia': f'{madu_akasia_prob * 100:.2f}%',
        #     'madu_hutan': f'{madu_hutan_prob * 100:.2f}%'
        # })

        return JsonResponse({
            'madu_akasia': round(madu_akasia_prob * 100, 6),
            'madu_hutan': round(madu_hutan_prob * 100, 6)
        })
    return JsonResponse({'error': 'Invalid request'}, status=400)
