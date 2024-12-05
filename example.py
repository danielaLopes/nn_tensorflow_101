import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


class CustomLayer(tf.keras.layers.Layer):
    """A custom TensorFlow layer that performs a linear transformation on input data.

    This layer implements a simple linear transformation by multiplying the input
    with a trainable weight matrix. The transformation is of the form: output = input × W,
    where W is the trainable weight matrix.

    Args:
        units (int): The number of output dimensions/neurons for this layer

    Attributes:
        w (tf.Variable): The trainable weight matrix created during build()
    """
    def __init__(self, units):
        super(CustomLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Args:
            input_shape: Shape tuple (tuple of integers) or TensorShape

        Returns:
            A TensorShape instance or tuple representing output shape
        """
        input_shape = tf.TensorShape(input_shape)
        # Output shape will be (batch_size, units)
        # We preserve the batch dimension and change the last dimension to self.units
        output_shape = input_shape[:-1].concatenate([self.units])
        return output_shape


class CustomLoss(tf.keras.losses.Loss):
    """Custom loss function implementing Mean Squared Error for classification tasks.

    This loss function converts sparse labels to one-hot encoding and then computes
    the mean squared difference between predictions and true labels. It properly
    inherits from tf.keras.losses.Loss for better integration with TensorFlow.

    Args:
        reduction (str): Type of reduction to apply to the loss.
            Options are 'auto', 'none', 'sum', or 'sum_over_batch_size'.
            Documented in https://www.tensorflow.org/api_docs/python/tf/keras/losses/Reduction
        name (str): Name of the loss function.
    """

    def __init__(self, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='custom_loss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        """Calculates the loss between true labels and predictions.

        Args:
            y_true (tf.Tensor): Ground truth labels in sparse format (class indices)
            y_pred (tf.Tensor): Model predictions in one-hot format

        Returns:
            tf.Tensor: The computed loss value
        """
        # Convert sparse labels to one-hot encoding
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])

        # Compute squared difference
        squared_difference = tf.square(y_true - y_pred)

        # Reduce mean across the class dimension
        return tf.reduce_mean(squared_difference, axis=-1)


class CustomCallback(tf.keras.callbacks.Callback):
    """Custom callback for monitoring training progress.

    This callback prints messages at the beginning and end of each training epoch,
    helping track training progress.

    Attributes:
        epoch (int): Tracks the current epoch number
    """
    def __init__(self):
        super(CustomCallback, self).__init__()
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nStarting Epoch {epoch}")
        self.epoch

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nFinishing Epoch {epoch}")


class CustomMetric(tf.keras.metrics.Metric):
    """Custom metric that tracks the mean absolute difference between predictions and labels.

    This metric maintains a running average of absolute differences between
    one-hot encoded true labels and model predictions.

    Attributes:
        accumulator (tf.Variable): Stores the running sum of absolute differences
    """
    def __init__(self, name='custom_metric', **kwargs):
        super(CustomMetric, self).__init__(name=name, **kwargs)
        self.accumulator = self.add_weight(name='acc', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert y_true to one-hot encoding to match y_pred shape and type
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=10)
        values = tf.abs(y_true - y_pred)
        self.accumulator.assign_add(tf.reduce_mean(values))

    def result(self):
        return self.accumulator

    def reset_state(self):
        self.accumulator.assign(0.)


@tf.keras.utils.register_keras_serializable(package='CustomOptimizer')
class CustomOptimizer(tf.keras.optimizers.Optimizer):
    """Custom optimizer implementing a combination of Adam and momentum optimization.

    This optimizer combines features from Adam (adaptive learning rates with momentum)
    and classical momentum updates. It maintains running averages of gradients and
    their squares, and optionally applies momentum to the updates.

    Args:
        learning_rate (float): The learning rate (default: 0.01)
        beta_1 (float): Exponential decay rate for first moment estimates (default: 0.9)
        beta_2 (float): Exponential decay rate for second moment estimates (default: 0.999)
        momentum (float): The momentum factor (default: 0.0)
        weight_decay (float, optional): Weight decay factor for regularization
        name (str): Name of the optimizer (default: "CustomOptimizer")

    Attributes:
        _momentums (list): First moment estimates for each variable
        _velocities (list): Second moment estimates for each variable
        _momentum_vars (list): Momentum variables if momentum > 0
    """
    def __init__(
            self,
            learning_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            momentum=0.0,
            weight_decay=None,
            name="CustomOptimizer",
            **kwargs
    ):
        """Initialize the optimizer.

        Args:
            learning_rate: Float or schedule, the learning rate
            beta_1: Float, exponential decay rate for first moment
            beta_2: Float, exponential decay rate for second moment
            momentum: Float, momentum factor
            weight_decay: Float or None, weight decay factor
            name: String, name of the optimizer
            **kwargs: Additional optimizer arguments
        """
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            name=name,
            **kwargs
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.momentum = momentum

    def build(self, var_list):
        """Initialize optimizer variables.

        Args:
            var_list: List of model variables
        """
        super().build(var_list)
        # Create slots for first moment, second moment and momentum
        self._momentums = []
        self._velocities = []
        if self.momentum > 0:
            self._momentum_vars = []

        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    reference_variable=var,
                    name="m",
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    reference_variable=var,
                    name="v",
                )
            )
            if self.momentum > 0:
                self._momentum_vars.append(
                    self.add_variable_from_reference(
                        model_variable=var,
                        name="momentum",
                    )
                )

    def update_step(self, gradient, variable, learning_rate):
        """Update step for the variable.

        Args:
            gradient: The gradient
            variable: The variable to update
            learning_rate: The current learning rate
        """
        if gradient is None:
            return

        # Get the current index of the variable
        var_idx = self._get_variable_index(variable)

        # Get the momentum and velocity for this variable
        m = self._momentums[var_idx]
        v = self._velocities[var_idx]

        # Cast the learning rate and gradient to match variable dtype
        lr = tf.keras.ops.cast(learning_rate, variable.dtype)
        gradient = tf.keras.ops.cast(gradient, variable.dtype)

        # Get beta values as tensors with matching dtype
        beta_1 = tf.keras.ops.cast(self.beta_1, variable.dtype)
        beta_2 = tf.keras.ops.cast(self.beta_2, variable.dtype)

        # Update first moment estimate
        self.assign(
            m,
            beta_1 * m + (1 - beta_1) * gradient
        )

        # Update second moment estimate
        self.assign(
            v,
            beta_2 * v + (1 - beta_2) * tf.keras.ops.square(gradient)
        )

        # Compute bias-corrected moments
        m_hat = m / (1 - tf.keras.ops.power(beta_1, tf.keras.ops.cast(self.iterations + 1, variable.dtype)))
        v_hat = v / (1 - tf.keras.ops.power(beta_2, tf.keras.ops.cast(self.iterations + 1, variable.dtype)))

        # Compute update
        update = m_hat / (tf.keras.ops.sqrt(v_hat) + 1e-8)

        # Apply momentum if configured
        if self.momentum > 0:
            momentum_var = self._momentum_vars[var_idx]
            momentum = tf.keras.src.ops.cast(self.momentum, variable.dtype)
            self.assign(
                momentum_var,
                momentum * momentum_var + update
            )
            update = momentum_var

        # Apply update to variable
        self.assign_sub(variable, lr * update)


@tf.keras.utils.register_keras_serializable(package='CustomModel')
class CustomModel(tf.keras.Model):
    """Custom neural network model with specialized training behavior.

    This model implements a neural network with a custom layer followed by
    two dense layers. It includes custom training logic with gradient clipping
    and specialized metrics tracking.

    Attributes:
        custom_layer (CustomLayer): Initial custom transformation layer
        dense1 (Dense): First dense layer with ReLU activation
        dense2 (Dense): Output layer with softmax activation
        loss_tracker (Mean): Tracks the training loss
        acc_metric (SparseCategoricalAccuracy): Tracks classification accuracy
    """
    def __init__(self, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.custom_layer = CustomLayer(units=32)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        self.optimizer = None
        self._loss_fn = None
        self._metrics = None

    def build(self, input_shape):
        self.custom_layer.build(input_shape)
        custom_output_shape = self.custom_layer.compute_output_shape(input_shape)
        self.dense1.build(custom_output_shape)
        intermediate_shape = self.dense1.compute_output_shape(custom_output_shape)
        self.dense2.build(intermediate_shape)
        self.built = True

    def call(self, inputs, training=False):
        """Forward pass of the model."""
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        x = self.custom_layer(inputs)
        x = self.dense1(x)
        return self.dense2(x)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([shape[0], 10])

    def train_step(self, data):
        x, y = data

        # Forward pass
        with tf.GradientTape() as tape:
            if self._call_has_training_arg:
                y_pred = self(x, training=True)
            else:
                y_pred = self(x)

            loss = self.compute_loss(
                x=x, y=y, y_pred=y_pred
            )
            self.loss_tracker.update_state(loss)
            if self.optimizer is not None:
                loss = self.optimizer.scale_loss(loss)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Clip gradients to prevent exploding gradients
        # Here, you can do what you want with the gradients
        tf.print("Gradients before gradient clipping", gradients, summarize=20)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        tf.print("Gradients after gradient clipping", gradients, summarize=20)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return self.compute_metrics(x, y, y_pred)

    def test_step(self, data):
        x, y = data
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)

        # Update metrics
        loss = self.compute_loss(
            x=x, y=y, y_pred=y_pred
        )
        self.loss_tracker.update_state(loss)
        return self.compute_metrics(x, y, y_pred)

    def predict_step(self, data):
        return self(data, training=False)

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_metric]


class CustomDataset(tf.keras.utils.PyDataset):
    """Custom dataset implementation using PyDataset for memory-efficient data handling.

    This dataset handles batch creation and optional shuffling of data,
    inheriting from PyDataset for better memory management and faster data loading.

    Args:
        x (np.ndarray): Input features or file paths
        y (np.ndarray): Target labels
        batch_size (int): Size of each batch (default: 32)
        shuffle (bool): Whether to shuffle data between epochs (default: True)
    """
    def __init__(self, x, y, batch_size=32, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """Returns the number of batches in the dataset."""
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        """Get a batch of data.

        Args:
            idx (int): Batch index

        Returns:
            tuple: (batch_x, batch_y) containing the batch data
        """
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.x))
        batch_indices = self.indices[start_idx:end_idx]

        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]

        return batch_x, batch_y

    def on_epoch_end(self):
        """Called at the end of every epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_synthetic_dataset(num_samples=1000, input_dim=20, num_classes=10):
    # Create synthetic features
    X = np.random.randn(num_samples, input_dim)

    # Create synthetic labels
    y = np.random.randint(0, num_classes, size=num_samples)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_custom_nn():
    X_train, X_test, y_train, y_test = create_synthetic_dataset()

    train_dataset = CustomDataset(X_train, y_train, batch_size=32)
    test_dataset = CustomDataset(X_test, y_test, batch_size=32, shuffle=False)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=artifacts_dir / 'best_model.keras',
        monitor='val_loss',
        save_best_only=True
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        min_lr=1e-6
    )

    model = CustomModel()
    optimizer = CustomOptimizer(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        momentum=0.0,
        weight_decay=0.01,
        clipnorm=1.0
    )
    model.compile(
        optimizer=optimizer,
        loss=CustomLoss(),
        metrics=[CustomMetric()]
    )
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=10,
        callbacks=[
            CustomCallback(),
            checkpoint_callback,
            early_stopping,
            lr_scheduler
        ])
    model.evaluate(test_dataset)
    print(f"Predictions before saving model: {model.predict(test_dataset)}")

    model_path = artifacts_dir / 'custom_model.keras'
    model.save(model_path)
    loaded_model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'CustomModel': CustomModel,
            'CustomLayer': CustomLayer,
            'CustomLoss': CustomLoss,
            'CustomMetric': CustomMetric,
            'CustomOptimizer': CustomOptimizer,
        },
    )

    loaded_model.summary()
    print(f"Predictions after saving model: {loaded_model.predict(test_dataset)}")


if __name__ == '__main__':
    train_custom_nn()
