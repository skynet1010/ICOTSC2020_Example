from Model.Layer import Layer


def create_model_vector_gru(output_length, output_features):
    model = [
        Layer(layer_type='GRU', size=2 * output_features, return_sequences=True),
        Layer(layer_type='GRU', size=4 * output_features, return_sequences=False),
        Layer(layer_type='Dropout', size=0.5),
        Layer(layer_type='Dense', size=output_features * output_length, activation='linear'),
        Layer(layer_type='Reshape', size=(output_length, output_features), activation='linear')
    ]
    return model
