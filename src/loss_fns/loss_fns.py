import tensorflow as tf


def create_loss_fns():
    bce = tf.keras.losses.BinaryCrossentropy()

    cce = tf.keras.losses.CategoricalCrossentropy()

    mse = tf.keras.losses.MeanSquaredError()

    mae = tf.keras.losses.MeanAbsoluteError()

    huber = tf.keras.losses.Huber()


if __name__ == '__main__':
    create_loss_fns()