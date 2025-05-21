# Import libraries
#import re
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template # make_response, jsonify,redirect,
import base64 


app = Flask(__name__)

# Load Model 
MODEL = tf.keras.models.load_model("./model/1.h5", compile=False)
'''
Earlier versions of Keras sometimes defaulted to reduction="auto" internally. 
Newer versions of Keras (especially from 2023 onward) have made this behavior 
stricter and removed support for "auto" as a valid value.
'''
MODEL.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
    metrics=['accuracy']
)

CLASS_Names = ["Negative", "Positive"] #'NORMAL', 'PNEUMONIA'


def read_file_as_image(data):  
    """
    Convert the uploaded file into a NumPy array for model prediction.
    """
    image = Image.open(BytesIO(data)).convert('RGB')
    # print(img_array)
    image = image.resize((256, 256))  # Adjust size based on your model's input (224, 224)
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image

def encode_image_base64(file):
    """
    Encode the uploaded image in Base64 format.
    """
    img_pil = Image.fromarray((file * 255).astype(np.uint8))
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/aboutUs', methods=['GET'])
def aboutUs():
   return render_template('aboutUs.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle the file upload and perform prediction.
    """
    if 'xray_image' not in request.files:
        return render_template("index.html", error="No file uploaded")

    file = request.files['xray_image']
    if file.filename == '':
        return render_template("index.html", error="No file selected")

    try:
        # Read the file as an image
        image = read_file_as_image(file.read())
        img_batch = np.expand_dims(image, 0)  # Expand dimensions for model input

        # Make predictions
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_Names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])*100

        # Encode the image in Base64 format
        encoded_image = encode_image_base64(image)

        # Render the template with prediction and image
        return render_template(
            "index.html",
            prediction=f"Person is Pneumonia {predicted_class} ({confidence:.2f}%)",
            image_data=f"data:image/png;base64,{encoded_image}"
        )

    except Exception as e:
        return render_template("index.html", error=f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

