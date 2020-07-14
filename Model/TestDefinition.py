class TestDefinition:
    def __init__(self, batch_size, sequence_length, prediction_length, loss, optimizer, steps_epoch, epoch, learning_rate=0.001, layer=[], model_name=None):
        self.layer = layer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.loss = loss
        self.optimizer = optimizer
        self.steps_epoch = steps_epoch
        self.epoch = epoch
        self.model_name = model_name
        self.learning_rate = learning_rate
