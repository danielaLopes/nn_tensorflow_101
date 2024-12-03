import tensorflow as tf


def create_layers():
    dense = tf.keras.layers.Dense(
        units=256,  # Number of neurons in the layer
        activation='relu'  # Activation function
    )

    convolutional = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu'
    )

    recurrent = tf.keras.layers.LSTM(
        units=32,
        return_sequences=False  # Only return final timestep
    )

    dropout = tf.keras.layers.Dropout(rate=0.5)

    batch_norm = tf.keras.layers.BatchNormalization()


if __name__ == '__main__':
    create_layers()