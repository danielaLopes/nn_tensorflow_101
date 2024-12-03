import tensorflow as tf


def create_initializations():
    glorot = tf.keras.initializers.GlorotUniform()

    he_normal = tf.keras.initializers.HeNormal()

    lecun = tf.keras.initializers.LecunNormal()

    custom = tf.keras.initializers.TruncatedNormal(
        mean=0.0,
        stddev=0.05
    )


if __name__ == '__main__':
    create_initializations()