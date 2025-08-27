# 🧠 Neural Network from Scratch (NumPy, No TF/Keras)

## 📌 Overview

This project implements a simple neural network from scratch using only NumPy and pure mathematics — without TensorFlow, PyTorch, or Keras.
The network is trained on the MNIST handwritten digit dataset (0–9) to demonstrate forward propagation, backpropagation, and gradient descent.
The goal is to understand the internals of a neural network by building every component step by step.

## ✨ Features

- ⚡ Forward and backward propagation implemented manually
- 🔢 Supports ReLU and Softmax activation functions
- 🌀 Gradient descent optimization without external ML libraries
- 🧩 One-hot encoding for labels
- 🖼️ Trains on the MNIST dataset
- 📈 Prints training accuracy during training

## 📂 Repository Structure

📦 NeuralNetwork_fromScratch/
┣ 📜 basic.py # Core implementation of the network
┣ 📜 README.md # Documentation
┗ 📜 requirements.txt # Dependencies (numpy, matplotlib, pandas)

## ⚙️ Installation

Clone the repository and install dependencies:
git clone <https://github.com/LeenaKannan/NeuralNetwork_fromScratch.git>
cd NeuralNetwork_fromScratch
pip install -r requirements.txt

## Train the model

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.1, iterations=500)

## 🔑 Code Explanation (Core Functions)

- 🏗️ `init_params()` → Initializes weights and biases randomly.
- 💡 `ReLU(Z)` → ReLU activation: replaces negatives with 0.
- 🎯 `softmax(Z)` → Converts raw scores into probabilities.
- 🔄 `forward_prop(...)` → Performs forward propagation.
- 🧮 `ReLU_deriv(Z)` → Derivative of ReLU for backprop.
- 🏷️ `one_hot(Y)` → Converts labels into one-hot vectors.
- 📤 `backward_prop(...)` → Computes gradients with backprop.
- 🔧 `update_params(...)` → Updates weights/biases (gradient descent).
- 📊 `get_predictions(...)` → Returns predicted class labels.
- ✅ `get_accuracy(...)` → Computes prediction accuracy.
- 🔁 `gradient_descent(...)` → Full training loop.

## 🧮 Mathematical Formulation

### 🔀 Forward Propagation

$$Z^{[1]} = W^{[1]}X + b^{[1]}, \quad A^{[1]} = ReLU(Z^{[1]})$$

$$Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}, \quad A^{[2]} = softmax(Z^{[2]})$$

### 🔄 Backward Propagation

$$dZ^{[2]} = A^{[2]} - Y, \quad dW^{[2]} = \frac{1}{m}dZ^{[2]}A^{[1]T}, \quad db^{[2]} = \frac{1}{m}\sum dZ^{[2]}$$

$$dZ^{[1]} = W^{[2]T}dZ^{[2]} \odot ReLU'(Z^{[1]}), \quad dW^{[1]} = \frac{1}{m}dZ^{[1]}X^T, \quad db^{[1]} = \frac{1}{m}\sum dZ^{[1]}$$

### ⚙️ Gradient Descent Update

$$W^{[l]} = W^{[l]} - \alpha dW^{[l]}, \quad b^{[l]} = b^{[l]} - \alpha db^{[l]}$$

## 📊 Training Setup (MNIST)

- 🖼️ Input size: 784 (28×28 pixels flattened)
- 🏗️ Hidden layer: 10 neurons with ReLU
- 🎯 Output layer: 10 neurons with Softmax (digits 0–9)
- ⚡ Optimizer: Gradient Descent

## 🤝 Contributing

Ideas for improvement:

- ➕ Add more hidden layers
- ⚡ Try advanced optimizers (Adam, RMSProp)
- 📉 Plot accuracy/loss curves
- 🔍 Add gradient checking tests

## 📜 License

📄 This project is licensed under the MIT License.
