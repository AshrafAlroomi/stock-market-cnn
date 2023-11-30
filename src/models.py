from keras.src.layers import Reshape, Dropout, Conv3D, MaxPooling3D
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model
from keras.utils import plot_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


class ModelTrainer:
    def __init__(self, model, epochs=100, batch_size=32):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

    def train_model(self, x_train, y_train, validation_data=None):
        history = self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1,
                                 validation_data=validation_data)
        return history


class UCNNModel:
    def __init__(self):
        self.model = Sequential()

    def build_model(self, input_shape, num_feature_maps, kernel_size, pool_size, dense_units):
        # Reshape layer to fit the expected input of Conv2D
        self.model.add(Reshape((input_shape[0], input_shape[1], 1), input_shape=(input_shape[0], input_shape[1], 1)))
        # First conv layer processing the entire feature set
        self.model.add(Conv2D(filters=num_feature_maps, kernel_size=(1, input_shape[1]), activation='relu'))
        # Second conv layer processing time-series data
        self.model.add(Conv2D(filters=num_feature_maps, kernel_size=(kernel_size, 1), activation='relu'))
        # Max pooling layer
        self.model.add(MaxPooling2D(pool_size=(pool_size, 1)))
        # Third convolutional layer with 8 filters of size (3, 1)
        self.model.add(Conv2D(num_feature_maps, (pool_size, 1), activation='relu'))
        # Second max pooling layer
        self.model.add(MaxPooling2D(pool_size=(pool_size, 1)))
        # Flatten layer to convert the 2D feature maps to a 1D vector
        self.model.add(Flatten())
        # Add dropout
        self.model.add(Dropout(0.2))
        # Fully connected layer
        self.model.add(Dense(dense_units, activation='relu'))
        # Output layer
        self.model.add(Dense(1))

    def build_model_3d(self, input_shape, num_feature_maps, kernel_size, pool_size, dense_units):
        self.model.add(Conv3D(filters=num_feature_maps, kernel_size=(kernel_size, kernel_size, kernel_size), activation='relu', padding='same', input_shape=input_shape))
        self.model.add(Conv3D(filters=num_feature_maps, kernel_size=(kernel_size, kernel_size, kernel_size), activation='relu', padding='same'))
        self.model.add(MaxPooling3D(pool_size=(pool_size, 1, pool_size), padding='same'))
        self.model.add(Conv3D(filters=num_feature_maps, kernel_size=(kernel_size, kernel_size, kernel_size), activation='relu', padding='same'))
        self.model.add(MaxPooling3D(pool_size=(pool_size, 1, pool_size), padding='same'))
        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=dense_units, activation='relu'))
        self.model.add(Dense(units=input_shape[1], activation='sigmoid'))  # Adjust the output units as needed

    def compile_model(self):
            self.model.compile(optimizer='adam', loss='mean_squared_error')
            return self.model

    def save_model(self, file_path):
        # Save the model to the specified file path
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        # Load and return the model from the specified file path
        self.model = load_model(file_path)
        print(f"Model loaded from {file_path}")
        return self.model

    def summary(self):
        # Print the model summary
        self.model.summary()

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")
        return mse, r2

    @staticmethod
    def plot_predictions(y_test, y_pred, title='Predictions vs Actual'):
        plt.figure(figsize=(10, 5))
        plt.plot(y_test, label='Actual')
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Target Variable')
        plt.legend()
        plt.show()
