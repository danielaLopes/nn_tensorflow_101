from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def create_plot_activation_fns():
    relu = tf.keras.activations.relu

    sigmoid = tf.keras.activations.sigmoid

    tanh = tf.keras.activations.tanh

    leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)

    selu = tf.keras.activations.selu

    elu = tf.keras.activations.elu

    softmax = tf.keras.activations.softmax


    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams['lines.linewidth'] = 3

    x = np.linspace(-5, 5, 200)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Common Neural Network Activation Functions', fontsize=16)

    ax1.plot(x, relu(x), 'b-', label='ReLU')
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, sigmoid(x), 'r-', label='Sigmoid')
    ax2.grid(True, alpha=0.3)

    ax3.plot(x, tanh(x), 'g-', label='Tanh')
    ax3.grid(True, alpha=0.3)

    ax4.plot(x, leaky_relu(x), 'c-', label='Leaky ReLU')
    ax4.grid(True, alpha=0.3)

    ax5.plot(x, selu(x), 'm-', label='SELU')
    ax5.grid(True, alpha=0.3)

    ax6.plot(x, elu(x), 'y-', label='ELU')
    ax6.grid(True, alpha=0.3)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_xlim([-5, 5])
        ax.set_ylim([-2, 2])
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(results_dir / "activation_functions.png")


    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    x = np.linspace(-10, 10, 200)

    inputs = np.vstack([x, x / 2, np.zeros_like(x)])  # Three different inputs
    probs = softmax(inputs, axis=0)  # Apply softmax across classes

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Nice color scheme
    labels = ['Class 1', 'Class 2', 'Class 3']

    for prob, color, label in zip(probs, colors, labels):
        ax.plot(x, prob, '-', color=color, label=label, linewidth=3)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('Input', fontsize=14, labelpad=10)
    ax.set_ylabel('Probability', fontsize=14, labelpad=10)
    ax.set_title('Softmax Function Behavior', fontsize=16, pad=20)

    ax.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

    ax.set_xlim([-10, 10])
    ax.set_ylim([-0.1, 1.1])

    plt.tight_layout()
    plt.savefig(results_dir / "softmax.png")


    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Understanding Softmax Function', fontsize=16)

    ax1 = plt.subplot(221)
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    inputs = np.stack([X, Y])
    softmax_output = tf.nn.softmax(inputs, axis=0)

    im1 = ax1.contourf(X, Y, softmax_output[0], levels=20, cmap='viridis')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('2D Softmax: Probability of Class 1')
    ax1.set_xlabel('Input 1')
    ax1.set_ylabel('Input 2')

    ax2 = plt.subplot(222)
    inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    temperatures = [0.1, 1.0, 2.0, 5.0]

    for temp in temperatures:
        scaled_inputs = inputs / temp
        softmax_outputs = tf.nn.softmax(scaled_inputs)
        ax2.plot(inputs, softmax_outputs, 'o-', label=f'T={temp}')

    ax2.set_title('Softmax with Different Temperatures')
    ax2.set_xlabel('Input Values')
    ax2.set_ylabel('Softmax Probabilities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(223)
    raw_scores = np.array([2.0, 1.0, 0.1, 3.0, 0.5])
    softmax_probs = tf.nn.softmax(raw_scores)

    x = np.arange(len(raw_scores))
    width = 0.35

    ax3.bar(x - width / 2, raw_scores, width, label='Raw Scores')
    ax3.bar(x + width / 2, softmax_probs, width, label='Softmax Probabilities')

    ax3.set_xticks(x)
    ax3.set_xticklabels(['Class ' + str(i + 1) for i in range(len(raw_scores))])
    ax3.legend()
    ax3.set_title('Raw Scores vs Softmax Probabilities')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(224)
    scores = np.array([2.0, 1.0, 0.1, 3.0, 0.5])
    exp_scores = np.exp(scores)
    softmax_probs = exp_scores / np.sum(exp_scores)

    steps = [scores, exp_scores, softmax_probs]
    labels = ['Original', 'After exp', 'After normalization']

    for i, (step_values, label) in enumerate(zip(steps, labels)):
        ax4.plot(range(len(scores)), step_values, 'o-', label=label)

    ax4.set_xticks(range(len(scores)))
    ax4.set_title('Softmax Transformation Steps')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "softmax_deep_dive.png")


if __name__ == '__main__':
    create_plot_activation_fns()
