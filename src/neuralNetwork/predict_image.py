import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model('mnist_model.keras')

def load_and_preprocess_image(file_path):
    try:
        img = Image.open(file_path).convert('L')
        original_size = img.size
        img = img.resize((28, 28))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        print(f"Loaded image '{file_path}' with original size {original_size} and resized to {img.shape[1:3]}")
        
        plt.imshow(img[0], cmap='gray')
        plt.title('Preprocessed Image')
        plt.show()
        
        return img
    except Exception as e:
        print(f"Error loading and preprocessing image: {e}")
        sys.exit(1)

def predict_image(file_path):
    img = load_and_preprocess_image(file_path)
    predictions = model.predict(img)
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions)
    print(f'Predicted digit: {predicted_label} with confidence: {confidence:.2f}')

if __name__ == "__main__":
    image_path = 'src/image/img.png'
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' not found.")
        sys.exit(1)
    predict_image(image_path)