import numpy as np
from sklearn.datasets import fetch_openml
from model import SimpleNeuralNetwork

print("Loading MNIST dataset..")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.to_numpy().astype(np.float32)
Y = mnist.target.to_numpy().astype(np.int32)

X /= 255

train_size = 5000

x_train, y_train = X[:train_size], Y[:train_size]
print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")

model = SimpleNeuralNetwork(input_size=784, output_size=10, hidden_size=64, seed=42)

#hyperparameters
# learning_rate = 0.1
# epochs = 10
# batch_size = 64

# num_samples = x_train.shape[0]

# for epoch in range(epochs):
#     indices = np.arange(num_samples)
#     np.random.default_rng(42+epoch).shuffle(indices)
#     X_shuffled = x_train[indices]
#     Y_shuffled = y_train[indices]

#     epoch_loss = 0.0
#     correct_predictions = 0

#     for i in range(0, num_samples, batch_size):

