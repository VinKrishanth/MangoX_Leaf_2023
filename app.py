from flask import Flask, render_template, request, send_from_directory
import cv2
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

model = tf.keras.models.load_model('models/mango_sequential_model.h5')

class_labels =['Anthracnose',
 'Bacterial Canker',
 'Cutting Weevil',
 'Die Back',
 'Gall Midge',
 'Healthy',
 'Powdery Mildew',
 'Sooty Mould']

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def generate_output_image(input_img, predictions):
    output_img = np.copy(input_img)

    predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
    color = (0, 255, 0)  
    thickness = 2

    height, width, _ = input_img.shape
    x_start = width // 4
    y_start = height // 4
    x_end = 3 * width // 4
    y_end = 3 * height // 4

    cv2.rectangle(output_img, (x_start, y_start), (x_end, y_end), color, thickness)

    return output_img

def resize_image(img, target_size=(256, 256)):
    return cv2.resize(img, target_size)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']

    input_image_filename = 'input.jpg'
    input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], input_image_filename)

    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    img = resize_image(img, target_size=(256, 256))

    cv2.imwrite(input_image_path, img)

    img = img / 255.0 

    predictions = model.predict(tf.expand_dims(img, axis=0))

    output_image = generate_output_image(img, predictions)
    output_image_filename = 'output.jpg'
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_image_filename)
    cv2.imwrite(output_image_path, output_image)

    result = {
        'prediction': class_labels[tf.argmax(predictions, axis=1).numpy()[0]],
        'input_image': 'uploads/input.jpg',
        'output_image': 'uploads/' + output_image_filename,
    }

    return render_template('result.html', prediction=result)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
