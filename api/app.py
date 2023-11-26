import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import os
from src.data_preprocessing import MultiSequencePipeline
from src.models import UCNNModel

app = Flask(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was transmitted
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = f'/tmp/{filename}'
        file.save(filepath)

        # Process the file and make predictions
        try:
            # Load the data
            data = pd.read_csv(filepath)
            processed_data = preprocess_data(data)

            # Make predictions
            print(processed_data.shape)
            predictions = model.predict(processed_data)

            # Convert predictions to a list for JSON response
            predictions_list = predictions.flatten().tolist()
            return jsonify({'predictions': predictions_list})

        except Exception as e:
            print(e)
            return jsonify({'error': 'Somethin went wrong'}), 500

    return jsonify({'error': 'Unsupported file type.'}), 400


def preprocess_data(data):
    from src.variables import Variables2D
    multi_sequence_pipeline = MultiSequencePipeline(data, Variables2D.INPUT_SHAPE[0])
    processed_data = multi_sequence_pipeline.process_for_prediction(data)
    return processed_data


def get_model(model_name):
    ucnn_model = UCNNModel()
    ucnn_model.load_model(model_name)
    return ucnn_model


model = get_model('../src/ucnn_model.h5')

if __name__ == '__main__':
    app.run(debug=True)
