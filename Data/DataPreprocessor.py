import pandas as pd
from sklearn import preprocessing
import numpy as np
import pickle


class DataPreprocessor:
    def __init__(self):
        self.data = pd.read_hdf('Data/sensordata.hd5', 'sensordata')
        self.num_sensors = len(self.data.columns) - 2

    @staticmethod
    def batch_generator(batch_size, sequence_length, prediction_length, step_size, x_data, y_data):
        idx = 0
        while True:
            x_shape = (batch_size, sequence_length, x_data.shape[1])
            x_batches = np.zeros(shape=x_shape, dtype=np.float32)
            y_shape = (batch_size, prediction_length, y_data.shape[1])
            y_batches = np.zeros(shape=y_shape, dtype=np.float32)

            for i in range(batch_size):
                pos = idx + i * step_size
                x_batches[i] = x_data[pos:pos + sequence_length]
                y_batches[i] = y_data[pos + sequence_length:pos + sequence_length + prediction_length]

            idx += batch_size * step_size
            if idx + prediction_length + batch_size * step_size + sequence_length > x_data.shape[0]:
                idx = 0
            yield (x_batches, y_batches)

    def get_data_from_generator(self, test_session, test_definition, window=None):
        data = self.data.copy()
        data.fillna(-1, inplace=True)

        if window is not None:
            data = data.resample(str(window) + 'min').mean().between_time('06:00', '22:00', include_end=False)

        test_data = data.values.astype('float32')

        x_scaler = preprocessing.RobustScaler()
        y_scaler = preprocessing.RobustScaler()
        scalerSettings = pickle.load(open("Data/scalerSettings.p", "rb"))
        x_scaler.scale_ = scalerSettings['xscale']
        x_scaler.center_ = scalerSettings['xcenter']
        y_scaler.scale_ = scalerSettings['yscale']
        y_scaler.center_ = scalerSettings['ycenter']

        test_x_scaled = x_scaler.transform(test_data)
        test_y_scaled = y_scaler.transform(test_data[:, 2:])

        test_gen = self.batch_generator(batch_size=test_x_scaled.shape[0] - test_definition.sequence_length - test_definition.prediction_length, sequence_length=test_definition.sequence_length, prediction_length=test_definition.prediction_length,
                                        step_size=1, x_data=test_x_scaled, y_data=test_y_scaled)
        test_x_scaled, test_y_scaled = next(test_gen)

        return test_x_scaled, test_y_scaled, y_scaler

    def get_data(self):
        return self.data.copy().fillna(-1)
