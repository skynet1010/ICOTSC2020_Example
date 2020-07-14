import tensorflow as tf
from datetime import datetime
import os


class ModelBuilder:
    def __init__(self):
        self.model_summary = ''

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

    def capture_summary(self, line):
        line = line.replace('======', '______').replace('=====', '_____')
        self.model_summary += line + '  \n'

    def get_model(self, test_session, test_definition, num_input_cols, metrics=None):
        model = tf.keras.models.Sequential()

        first = True
        for tl in test_definition.layer:
            kwargs = {}
            if first:
                kwargs['input_shape'] = (test_definition.sequence_length, num_input_cols)

            if tl.layer_type == "GRU":
                if tf.test.is_gpu_available(cuda_only=True):
                    model.add(tf.keras.layers.CuDNNGRU(units=tl.size, return_sequences=tl.return_sequences, kernel_regularizer=tf.keras.regularizers.l2(1e-3), **kwargs))
                else:
                    raise ValueError('Trained model does only support GPU/CUDA version of GRU layer.')
            elif tl.layer_type == "Dense":
                model.add(tf.keras.layers.Dense(units=tl.size, activation=tl.activation, **kwargs))
            elif tl.layer_type == "Time_Dense":
                model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=tl.size, activation=tl.activation, **kwargs)))
            elif tl.layer_type == "Conv1D":
                model.add(tf.keras.layers.Conv1D(filters=tl.size[0], kernel_size=tl.size[1], padding='same', activation=tl.activation, kernel_regularizer=tf.keras.regularizers.l2(1e-3), **kwargs))
            elif tl.layer_type == "MaxPooling1D":
                model.add(tf.keras.layers.MaxPool1D(pool_size=tl.size, **kwargs))
            elif tl.layer_type == "Dropout":
                model.add(tf.keras.layers.Dropout(rate=tl.size, **kwargs))
            elif tl.layer_type == "SpatialDropout1D":
                model.add(tf.keras.layers.SpatialDropout1D(rate=tl.size, **kwargs))
            elif tl.layer_type == "Flatten":
                model.add(tf.keras.layers.Flatten(**kwargs))
            elif tl.layer_type == "Reshape":
                model.add(tf.keras.layers.Reshape(target_shape=tl.size, **kwargs))
            elif tl.layer_type == "Repeat":
                model.add(tf.keras.layers.RepeatVector(tl.size, **kwargs))
            elif tl.layer_type == "BatchNormalization":
                model.add(tf.keras.layers.BatchNormalization(**kwargs))
            else:
                raise ValueError('Wrong layer type')

            first = False

        if test_definition.loss == 'MSE':
            loss = tf.keras.losses.MSE
        elif test_definition.loss == 'RMSE':
            loss = self.root_mean_squared_error
        else:
            raise ValueError('Wrong loss type')

        if test_definition.optimizer == 'Adam':
            optimizer = tf.keras.optimizers.Adam(lr=test_definition.learning_rate)
        else:
            raise ValueError('Wrong optimizers type')

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.summary(print_fn=self.capture_summary)

        self.model_summary += '  \n' + \
                              'model name: ' + str(test_definition.model_name) + '\n' + \
                              'resample_window: ' + str(test_session.resample_window) + '\n' + \
                              'batch_size: ' + str(test_definition.batch_size) + '\n' + \
                              'sequence_length: ' + str(test_definition.sequence_length) + '\n' + \
                              'prediction_length: ' + str(test_definition.prediction_length) + '\n' + \
                              'loss: ' + test_definition.loss + '\n' + \
                              'optimizer: ' + test_definition.optimizer + '\n' + \
                              'steps_epoch: ' + str(test_definition.steps_epoch) + '\n' + \
                              'epoch: ' + str(test_definition.epoch) + '\n'

        checkpoint_path = 'Model/model_checkpoint.keras'

        return model, self.model_summary, checkpoint_path
