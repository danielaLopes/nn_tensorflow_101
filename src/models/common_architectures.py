import tensorflow as tf


def create_ffn(input_shape=(784,), layers=[64, 32], activation='relu'):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        *[tf.keras.layers.Dense(units, activation=activation)
          for units in layers],
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_cnn(input_shape=(28, 28, 1)):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu'
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu'
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])


def create_rnn(input_shape=(100, 300), rnn_units=128, dense_layers=[64]):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(rnn_units, return_sequences=False),
        tf.keras.layers.BatchNormalization(),
        *[tf.keras.layers.Dense(units, activation='relu')
          for units in dense_layers],
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_transformer(input_shape=(100,), embed_dim=256, num_heads=8, ff_dim=512):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Embedding
    x = tf.keras.layers.Embedding(input_dim=10000, output_dim=embed_dim)(inputs)
    x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.LayerNormalization()(x)


    # Feed forward
    ff = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ff = tf.keras.layers.Dense(embed_dim)(ff)
    ff = tf.keras.layers.Dropout(0.1)(ff)
    x = tf.keras.layers.LayerNormalization()(x + ff)

    # Global pooling and output
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)


def create_autoencoder(input_shape=(784,), encoding_dim=32):
    # Encoder
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(encoding_dim, activation='relu')
    ])

    # Decoder
    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(input_shape[0], activation='sigmoid')
    ])

    # Full autoencoder
    autoencoder = tf.keras.Sequential([encoder, decoder])
    return autoencoder


if __name__ == '__main__':
    create_ffn()
    create_cnn()
    create_rnn()
    create_autoencoder()
    create_transformer()
