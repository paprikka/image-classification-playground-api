import logging
from fastai.vision.all import PILImage
from image_classification_crows import model
from flask import Flask, request
from fastai import *

app = Flask(__name__)


@app.route("/foo")
def health():
    return "oi hi its me temme"


@app.route("/predict", methods=['POST'])
def predict():
    try:
        image_payload = request.files['image']
        image = PILImage.create(image_payload)
        result = model.classify_image(image)
        print(result)
        return {
            'predictionResult': result
        }
    except:
        logging.exception("Prediction error")
        return {'error': 'Prediction Error'}, 500


if __name__ == '__main__':
    app.run(debug=True, port=3000)
