import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.variables import ALL_COLS


class PreprocessingPipeline:
    def __init__(self, data):
        self.data = data

    def handle(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def next(self, next_step):
        return next_step(self.data)


class HandleNAValues(PreprocessingPipeline):
    def handle(self, *args, **kwargs):
        # Fill missing values with forward fill for specified columns
        for col in kwargs.get('fillna_cols', []):
            self.data[col].fillna(method='ffill', inplace=True)
        # Drop remaining NaNs
        self.data.dropna(inplace=True)
        assert self.data.isnull().sum().sum() == 0
        return self


class ScaleFeatures(PreprocessingPipeline):
    def handle(self, *args, **kwargs):
        features = self.data.drop(columns='Target', errors='ignore')
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        self.data.loc[:, features.columns] = scaled_features
        return self


class GenerateY(PreprocessingPipeline):
    def handle(self, *args, **kwargs):
        self.data['Target'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)
        self.data.dropna(inplace=True)
        return self


class DropUnwantedCols(PreprocessingPipeline):
    def handle(self, *args, **kwargs):
        drop_cols = kwargs.get('drop_cols', [])
        self.data.drop(columns=drop_cols, inplace=True, errors='ignore')
        return self


class FeatureReduction(PreprocessingPipeline):
    def handle(self, *args, **kwargs):
        x = self.data.drop("Target", axis=1)
        y = self.data["Target"]
        pca = PCA(n_components=0.95)
        x_pca = pca.fit_transform(x)
        pca_columns = [f'PC{i + 1}' for i in range(x_pca.shape[1])]
        self.data = pd.DataFrame(x_pca, columns=pca_columns)
        self.data['Target'] = y.values
        return self


class TrainTestSplit(PreprocessingPipeline):
    def handle(self, *args, **kwargs):
        test_size = kwargs.get("test_size", 0.2)
        shuffle = kwargs.get("shuffle", False)
        x = self.data.drop("Target", axis=1)
        y = self.data["Target"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=shuffle)
        return x_train, x_test, y_train, y_test


class AlignColumns(PreprocessingPipeline):
    def __init__(self, data):
        super().__init__(data)

    def handle(self):
        missing_cols = set(ALL_COLS) - set(self.data.columns)
        for col in missing_cols:
            self.data[col] = 0
        # Ensure columns are in the same order for all dataframes
        self.data = self.data.reindex(columns=ALL_COLS, fill_value=0)
        return self


# class AddPadding(PreprocessingPipeline):
#     def handle(self, *args, **kwargs):
#         num_timesteps = kwargs.get('num_timesteps', 60)
#         num_features = self.data.shape[1]
#         padding_required = (num_timesteps - num_features % num_timesteps) % num_timesteps
#
#         # Create a DataFrame with zeros for padding
#         if padding_required > 0:
#             padding_df = pd.DataFrame(0, index=range(padding_required), columns=self.data.columns)
#             self.data = pd.concat([padding_df, self.data], ignore_index=True)
#
#         return self
#


class MultiSequencePipeline:
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def process_all(self):
        processed_data = {"train": {}, "test": {}}
        for key, df in self.data.items():
            x_train, x_test, y_train, y_test = self.preprocess_data(df)
            processed_data["train"][key] = self.process_single(x_train, y_train)
            processed_data["test"][key] = self.process_single(x_test, y_test)
        return processed_data

    def process_single(self, df, target):
        sequences, targets = self.transform(df, target)
        return {"x": sequences.reshape(-1, self.sequence_length, sequences.shape[2], 1),
                "y": targets}

    def transform(self, df, target):
        sequences = [df.iloc[i:i + self.sequence_length].values for i in range(len(df) - self.sequence_length)]
        targets = target[self.sequence_length:]
        return np.array(sequences), np.array(targets)

    def transform_for_prediction(self, df):
        sequences = [df.iloc[i:i + self.sequence_length].values for i in range(len(df) - self.sequence_length + 1)]
        return np.array(sequences).reshape(-1, self.sequence_length, df.shape[1], 1)

    def process_for_prediction(self, df):
        processed_df = self.preprocess_for_prediction(df)
        sequences = self.transform_for_prediction(processed_df)
        return sequences

    @staticmethod
    def preprocess_for_prediction(df):
        pipeline = PreprocessingPipeline(df)
        processed_data = (
            pipeline
            .next(AlignColumns).handle()
            .next(DropUnwantedCols).handle(drop_cols=['Date', 'Name', 'XAG', 'XAU'])
            .next(HandleNAValues).handle(fillna_cols=['CAD', 'JPY', 'CNY', 'GBP'])
            .next(ScaleFeatures).handle()
        )
        return processed_data.data

    @staticmethod
    def preprocess_data(df):
        pipeline = PreprocessingPipeline(df)
        x_train, x_test, y_train, y_test = (
            pipeline
            .next(AlignColumns).handle()
            .next(DropUnwantedCols).handle(drop_cols=['Date', 'Name', 'XAG', 'XAU'])
            .next(HandleNAValues).handle(fillna_cols=['CAD', 'JPY', 'CNY', 'GBP'])
            .next(GenerateY).handle()
            .next(ScaleFeatures).handle()
            # .next(FeatureReduction).handle()
            .next(TrainTestSplit).handle(test_size=0.2)
        )
        return x_train, x_test, y_train, y_test

    @staticmethod
    def merge(processed_data):
        # TODO MAKE IT SIMPLE
        x_train_list = []
        x_test_list = []

        y_train_list = []
        y_test_list = []
        for key, item in processed_data["train"].items():
            x_train_list.append(item["x"])
            y_train_list.append(item["y"])

        for key, item in processed_data["test"].items():
            x_test_list.append(item["x"])
            y_test_list.append(item["y"])

        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        x_test = np.concatenate(x_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        return x_train, x_test, y_train, y_test

# for testing
# if __name__ == '__main__':
#     sequence_length = 60
#     df_stock1 = pd.read_csv("../data/raw/Processed_S&P.csv")
#     df_stock2 = pd.read_csv("../data/raw/Processed_NASDAQ.csv")
#     data = {
#         'stock1': df_stock1,
#         'stock2': df_stock2,
#     }
#
#     multi_sequence_pipeline = MultiSequencePipeline(data, sequence_length)
#
#     processed_data = multi_sequence_pipeline.process_all()
