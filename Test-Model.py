from Data.DataPreprocessor import DataPreprocessor
from Model.TestSession import TestSession
from Model.TestDefinition import TestDefinition
from Model.VectorOutput import create_model_vector_gru

dp = DataPreprocessor()

sess = TestSession(name='Test', resample_window=30, data_preprocessor=dp)
sess.tests.append(TestDefinition(batch_size=128, sequence_length=200, prediction_length=20, loss='MSE', optimizer='Adam', steps_epoch=200,
                                 epoch=300, layer=create_model_vector_gru(20, dp.num_sensors), model_name='VectorOutput_GRU'))
sess.run()
