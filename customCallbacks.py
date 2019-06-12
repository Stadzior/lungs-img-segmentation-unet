import keras

class AccuracyAndLossCallback(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        self.loss = logs.get('loss')
        self.accuracy = logs.get('acc')