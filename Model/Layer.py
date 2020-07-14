class Layer:
    def __init__(self, layer_type, size=None, activation=None, return_sequences=True):
        self.layer_type = layer_type
        self.size = size
        self.activation = activation
        self.return_sequences = return_sequences
