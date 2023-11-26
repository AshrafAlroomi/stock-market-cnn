import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.data_preprocessing import MultiSequencePipeline
from src.models import UCNNModel, ModelTrainer
from src.variables import Variables3D

TRAIN = False

data_dir = "data/raw/"
model_name = 'src/3d.h5'


def read_data():
    dataframes = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            path = os.path.join(data_dir, filename)
            df = pd.read_csv(path)
            identifier = filename.split('.')[0].split('_')[1]
            dataframes[identifier] = df
    return dataframes


if __name__ == '__main__':
    dfs = read_data()
    multi_sequence_pipeline = MultiSequencePipeline(dfs, Variables3D.INPUT_SHAPE[0])
    processed_data = multi_sequence_pipeline.process_all()

    x_train, x_test, y_train, y_test = multi_sequence_pipeline.merge_3D(processed_data)

    ucnn_model = UCNNModel()
    ucnn_model.build_model_3d(input_shape=Variables3D.INPUT_SHAPE, num_feature_maps=Variables3D.NUM_OF_FEATURE_MAP,
                              kernel_size=Variables3D.KERNEL_SIZE, pool_size=Variables3D.POOL_SIZE,
                              dense_units=Variables3D.DENSE_UNITS)

    ucnn_model.compile_model()
    trainer = ModelTrainer(model=ucnn_model.model, epochs=100, batch_size=32)

    # Train the model
    if TRAIN:
        history = trainer.train_model(x_train=x_train, y_train=y_train, validation_data=(x_test, y_test))
        ucnn_model.save_model(model_name)
    else:
        ucnn_model.load_model(model_name)

    y_pred = ucnn_model.predict(x_test)
    y_pred_classes = (y_pred > 0.5).astype("int32")
    ucnn_model.evaluate(x_test, y_test)
    n = 100
    ucnn_model.plot_predictions(y_test[:n], y_pred[:n])

    accuracy = accuracy_score(y_test, y_pred_classes)
    f1 = f1_score(y_test, y_pred_classes, average='samples')
    precision = precision_score(y_test, y_pred_classes, average='samples')
    recall = recall_score(y_test, y_pred_classes, average='samples')

    # Print the computed metrics
    print(f"F1 Score (Samples): {f1:.4f}")
    print(f"Precision (Samples): {precision:.4f}")
    print(f"Recall (Samples): {recall:.4f}")
