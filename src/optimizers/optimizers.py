import tensorflow as tf


def create_optimizers():
    adam = tf.keras.optimizers.Adam(
        learning_rate=0.001,  # How big steps to take during optimization
        beta_1=0.9,  # First moment decay rate
        beta_2=0.999  # Second moment decay rate
    )

    sgd = tf.keras.optimizers.SGD(
        learning_rate=0.01,  # Usually needs higher values than Adam
        momentum=0.9,  # How much to consider previous updates
        nesterov=True  # Looks ahead before making updates, often improves convergence
    )

    adamw = tf.keras.optimizers.AdamW(
        learning_rate=0.001,  # Similar to Adam
        weight_decay=0.004  # Directly penalizes large weights (better regularization than L2)
    )

    adagrad = tf.keras.optimizers.Adagrad(
        learning_rate=0.01  # Initial learning rate
    )

    rmsprop = tf.keras.optimizers.RMSprop(
        learning_rate=0.001,  # Similar to Adam
        rho=0.9  # How quickly to forget old squared gradients
    )


if __name__ == '__main__':
    create_optimizers()