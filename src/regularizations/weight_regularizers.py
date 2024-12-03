import tensorflow as tf


def create_weight_regularizers():
    l1 = tf.keras.regularizers.L1(l1=0.01)

    l2 = tf.keras.regularizers.L2(l2=0.01)

    l1_l2 = tf.keras.regularizers.L1L2(l1=0.01, l2=0.01)


if __name__ == '__main__':
    create_weight_regularizers()