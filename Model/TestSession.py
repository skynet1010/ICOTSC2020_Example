import tensorflow as tf
import numpy as np
from sklearn import metrics
from Model.ModelBuilder import ModelBuilder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TestSession:
    def __init__(self, name, resample_window, data_preprocessor):
        self.tests = []
        self.name = name
        self.resample_window = resample_window
        self.data_preprocessor = data_preprocessor

    def run_test(self, test_definition):
        test_x_scaled, test_y_scaled, y_scaler = self.data_preprocessor.get_data_from_generator(self, test_definition, self.resample_window)
        model, model_summary, checkpoint_path = ModelBuilder().get_model(test_session=self, test_definition=test_definition, num_input_cols=test_x_scaled.shape[2], metrics=[ModelBuilder.root_mean_squared_error])
        print(model_summary)

        model.load_weights(checkpoint_path)
        test_pred = model.predict(test_x_scaled)
        test_y_rescaled = y_scaler.inverse_transform(test_y_scaled.reshape(-1, test_y_scaled.shape[2])).reshape(test_y_scaled.shape)
        test_pred_rescaled = y_scaler.inverse_transform(test_pred.reshape(-1, test_pred.shape[2])).reshape(test_pred.shape)

        rmse_step = np.sqrt(metrics.mean_squared_error(np.reshape(np.swapaxes(test_y_rescaled, 1, 2), (-1, test_definition.prediction_length)), np.reshape(np.swapaxes(test_pred_rescaled, 1, 2), (-1, test_definition.prediction_length)), multioutput='raw_values'))
        rmse_total = np.sqrt(metrics.mean_squared_error(np.reshape(np.swapaxes(test_y_rescaled,  1, 2), (-1, test_definition.prediction_length)), np.reshape(np.swapaxes(test_pred_rescaled,  1, 2), (-1, test_definition.prediction_length))))

        print('RMSE total: ' + str(rmse_total))
        print('RMSE per forecasting step: ' + str(rmse_step))

        tf.keras.backend.clear_session()

    def run(self):
        for td in self.tests:
            self.run_test(td)
