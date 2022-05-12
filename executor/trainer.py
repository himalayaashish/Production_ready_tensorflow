import os
import tensorflow as tf
from utils.logger import get_logger
from tensorflow.keras.optimizers import RMSprop
LOG = get_logger('trainer')


class Trainer:

    def __init__(self, model, X_train, Y_train, X_test, Y_test, batch_size, metric, epoches,):
        self.model = model
        self.batch_size = batch_size
        self.metric = metric
        self.epoches = epoches
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.log_dir = 'Classification_dir'
        self.callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)

        self.model_save_path = 'saved_models/'


    def train(self):
        self.model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(),
                      metrics=self.metric)
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        print(self.X_train.shape)
        print(self.X_test.shape)
        print(self.Y_train.shape)
        print(self.Y_test.shape)
        history = self.model.fit(self.X_train, self.Y_train,
                            batch_size=self.batch_size,
                            epochs=self.epoches,
                            validation_data=(self.X_test, self.Y_test))
        # ,callbacks=[tensorboard_callback])


        save_path = os.path.join(self.model_save_path, "click_model/1/")
        tf.saved_model.save(self.model, save_path)
        # tensorboard --logdir logs/gradient_tape