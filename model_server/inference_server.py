import logging
from flask import Flask, request
import numpy as np
import pickle


app = Flask(__name__)


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ],
    )


@app.before_first_request
def load_model():
    '''
    Load the trained model from the specified path.

    Return the trained model.
    '''
    model_path = "models/best_model_ever.pkl"
    logging.info(f"Loading model in path: {model_path}")
    with open(model_path, 'rb') as f:
        MODEL = pickle.load(f)
    logging.info(f"Model loaded")
    return MODEL


@app.route('/')
def home() -> str:
    return "Hello world from the inference server."


@app.route("/predict", methods=['POST'])
def predict() -> str:
    return "The predicted house price is: 5,000,000"


if __name__ == "__main__":
    setup_logger()
    MODEL = load_model()
    app.run(host='0.0.0.0', port = "5050", debug = True)