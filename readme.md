# Stock Market Prediction Project

## Overview

This project is designed to predict stock market trends using a Convolutional Neural Network (CNN) model. It includes a preprocessing pipeline to prepare the data and a Flask application to serve the model's predictions over an API.

## Model Description

The CNN model is trained on historical stock market data and predicts whether the stock closing price will go up or down. The model architecture is designed to work with time-series data, capturing temporal dependencies and patterns.

## Preprocessing Pipeline

The data preprocessing is handled by a `PreprocessingPipeline` class, which performs steps like handling missing values, scaling features, generating target variables, aligning columns, and more, to prepare the data for training and predictions.

## Flask API

The Flask API allows users to make predictions by submitting a CSV file containing the required features. The API processes the file, uses the model to predict stock trends, and returns the predictions.

## Installation

To set up the project environment:

```bash
# Clone the repository
git clone https://github.com/AshrafAlroomi/stock-market-cnn.git
cd stock-market-cnn

# Install the required dependencies
pip install -r requirements.txt
```
## Usage
### Jupyter Notebook
To interact with the data and preprocessing pipeline using a Jupyter notebook:

Start the Jupyter Notebook server:

```bash
jupyter notebook
```
- Open the jupyter/data_analysis.ipynb notebook.
- Run the notebook cells sequentially to see the data details.


### Flask API
- Navigate to the Flask application's directory.
- Start the Flask server:

```bash
flask run
```
The server will start on http://localhost:5000. Use this base URL to interact with the API.

### Making Predictions via API
```bash
curl -X POST -F 'file=@/path/to/your/csvfile.csv' http://localhost:5000/predict
```
The API will return a JSON response containing the predictions.


### Retrain the model
- open scr/main.py
```python
# Change 
TRAIN = True
```
```bash
python main.py
```



