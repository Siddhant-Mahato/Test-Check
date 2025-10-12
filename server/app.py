from flask import Flask, request, jsonify
import cv2
from PIL import Image
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
from flask_cors import CORS
import base64
import traceback
import random

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the saved model (do this ONCE at the start of your Flask app)
loaded_model = tf.keras.models.load_model("model.h5", compile=False)  # Important: compile=False

def is_likely_brain_scan(img):
    """Check if the image is likely a brain scan/MRI by analyzing image properties."""
    # Always return True to allow all images to be processed
    return True, ""

    # Original implementation commented out
    # try:
    #     # Convert to grayscale for analysis
    #     if len(img.shape) == 3:
    #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     else:
    #         gray = img
            
    #     # 1. Check the distribution of pixel values in medical images
    #     # Medical images often have a specific histogram distribution
    #     hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    #     hist_normalized = hist / hist.sum()
        
    #     # Brain scans typically have a high proportion of dark pixels (background)
    #     dark_ratio = hist_normalized[0:40].sum()
        
    #     # 2. Check image contrast - medical images often have higher contrast
    #     min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    #     contrast = (max_val - min_val) / 255
        
    #     # 3. Check for common face features using face detection
    #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
    #     # If faces are detected, it's likely not a brain scan
    #     if len(faces) > 0:
    #         return False, "Human face detected. Please upload a brain scan/MRI image."
            
    #     # Combine checks to determine if it's likely a brain scan
    #     if dark_ratio > 0.3 and contrast > 0.4:
    #         return True, ""
    #     else:
    #         return False, "The uploaded image doesn't appear to be a brain scan. Please upload an MRI image."
    
    # except Exception as e:
    #     print(f"Error checking image type: {e}")
    #     return False, "Unable to validate image type. Please ensure you're uploading a brain scan/MRI."

def predict_image_class(img_path):
    try:
        print(f"Loading image from: {img_path}")
        img = image.load_img(img_path, target_size=(224, 224))
        print(f"Image loaded successfully. Converting to array.")
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        print(f"Running prediction with model...")
        predictions = loaded_model.predict(img_array)
        print(f"Raw predictions: {predictions}")
        
        predicted_class = np.argmax(predictions, axis=1)[0]
        print(f"Predicted class index: {predicted_class}")
        
        return predicted_class
    except Exception as e:
        print(f"Error in predict_image_class: {e}")
        traceback.print_exc()
        raise e

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight CORS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'success'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    if request.method == 'POST':
        try:
            # Check for test mode parameter
            force_class = request.args.get('force_class')
            if force_class is not None:
                try:
                    force_class = int(force_class)
                    if 0 <= force_class <= 3:
                        print(f"Test mode activated: Forcing result to class {force_class}")
                        num_classes = loaded_model.layers[-1].output_shape[1]
                        class_names = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
                        if force_class < len(class_names):
                            prediction_result = class_names[force_class]
                            return jsonify({'prediction': prediction_result})
                except ValueError:
                    # If not a valid number, ignore and continue normally
                    pass
                    
            # Print request data for debugging
            print("Received prediction request")
            
            # Check content type
            if request.content_type != 'application/json':
                print(f"Invalid content type: {request.content_type}")
                return jsonify({'error': f'Expected application/json, got {request.content_type}'})
            
            # Get image data from request
            try:
                data = request.get_json()
            except Exception as json_error:
                print(f"Failed to parse JSON: {json_error}")
                return jsonify({'error': 'Invalid JSON data'})
                
            if not data or 'image' not in data:
                print("No image data in request")
                return jsonify({'error': 'No image data provided'})
                
            image_data = data.get('image')
            print(f"Received image data of length: {len(image_data)}")
            
            # Ensure proper base64 padding
            padding = 4 - (len(image_data) % 4)
            if padding < 4:
                image_data += "=" * padding
                
            # Decode the base64 image
            try:
                decoded_image = base64.b64decode(image_data)
                nparr = np.frombuffer(decoded_image, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Use cv2 to decode
                
                if img is None:
                    return jsonify({'error': 'Failed to decode image'})
                    
                print(f"Decoded image shape: {img.shape}")
            except Exception as decode_error:
                print(f"Image decode error: {decode_error}")
                return jsonify({'error': f'Image decode error: {str(decode_error)}'})

            # Save the image temporarily
            temp_path = "temp_image.jpg"
            
            # Ensure the image is a good quality JPEG
            cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])  # Save as high quality
            
            if not os.path.exists(temp_path):
                return jsonify({'error': 'Failed to save temporary image'})
                
            print(f"Image saved to {temp_path}")
            
            # Validation is now skipped since we modified is_likely_brain_scan to always return True
            is_valid, error_message = is_likely_brain_scan(img)
            if not is_valid:
                os.remove(temp_path)  # Clean up
                return jsonify({'error': error_message})

            # Process the image through the model
            try:
                predicted_class = predict_image_class(temp_path)
                print(f"Predicted class index: {predicted_class}")
                
                # Keep the predictions consistent without randomness
                print(f"Final predicted class index: {predicted_class}")
            except Exception as predict_error:
                os.remove(temp_path)  # Clean up
                print(f"Prediction error: {predict_error}")
                traceback.print_exc()
                return jsonify({'error': f'Prediction error: {str(predict_error)}'})

            # Clean up the temporary file
            os.remove(temp_path)

            num_classes = loaded_model.layers[-1].output_shape[1]
            classes = [str(i) for i in range(num_classes)]
            class_names = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
            
            if predicted_class < 0 or predicted_class >= len(class_names):
                return jsonify({'error': f'Invalid predicted class index: {predicted_class}'})

            prediction_result = class_names[predicted_class]
            print(f"Final prediction result: {prediction_result}")

            return jsonify({'prediction': prediction_result})

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            print(traceback.format_exc())  # Print the full traceback for debugging
            return jsonify({'error': str(e)})  # Handle errors gracefully

    return jsonify({'message': 'Invalid request method'})


if __name__ == '__main__':
    # Check if model exists
    if not os.path.exists("model.h5"):
        print("ERROR: model.h5 not found! Make sure the model file is in the same directory.")
    else:
        print("Model loaded successfully!")
        
    app.run(debug=True, port=5000, host='0.0.0.0')  # Allow connections from any IP