import tensorflow as tf


def dropout_techniques():
    model_standard_dropout = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # 50% dropout rate
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3)  # 30% dropout rate
    ])

    model_spatial_dropout = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.SpatialDropout2D(0.3),  # Drops 30% of feature maps
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu')
    ])

    model_recurrent_dropout = tf.keras.layers.LSTM(
        units=64,
        dropout=0.2,  # Dropout for inputs
        recurrent_dropout=0.2  # Dropout for recurrent connections
    )


if __name__ == '__main__':
    dropout_techniques()