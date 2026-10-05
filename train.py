import numpy as np
import pickle
import gzip
from model import SimpleNeuralNetwork

def load_data():
    filepath = 'data/mnist.pkl.gz'
    with gzip.open(filepath, "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
        x_train, y_train = training_data
        x_test, y_test = test_data
        X_val, Y_val = validation_data

        return x_train, x_test, y_train, y_test



def one_hot_encode(labels, num_classes=10):
    """
    converts 1D array of labels into 2D matrix with binary elements representing 
    the state of each class (also called one-hot-encoded matrix) as it gives 
    categorical data a proper mathematical form
    labels: (num_samples,) -> (num_samples, 10)
    """
    return np.eye(num_classes)[labels]


def compute_loss(y_pred, y_true_one_hot):
    """
    Computes cross-entropy-loss
    y_pred: (batch_size, 10) - predicted probs from softmax
    y_true: (batch_size, 10) - true values
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)

    loss = -np.mean(np.sum(y_true_one_hot * np.log(y_pred), axis=1))
    return loss

def compute_accuracy(y_pred, y_true_labels):
    """
    y_pred: (batchsize, 10) predicted probabilities
    y_true_labels: (batchsize,) true labels ei
    """
    predictions=np.argmax(y_pred, axis=1)
    accuracy = np.mean(predictions == y_true_labels)
    return accuracy

def train():
    epochs = 10
    batch_size = 64
    learning_rate = 0.1
    hidden_size = 64

    x_train, x_test, y_train, y_test = load_data()
    y_train_encoded = one_hot_encode(y_train)

    model = SimpleNeuralNetwork()

    num_samples = x_train.shape[0]
    num_batches = num_samples//batch_size

    print(f"starting training for {epochs} epoch\n")

    for epoch in range(1, epochs+1):
        indices = np.random.permutation(num_samples)
        x_shuffled = x_train[indices]
        y_shuffled = y_train_encoded[indices]

        running_loss = 0.0

        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = start_idx + batch_size

            x_batch = x_shuffled[start_idx, end_idx]
            y_batch = y_shuffled[start_idx, end_idx]

            probs = model.Forward(x_batch)

            loss = compute_loss(probs, y_batch)
            running_loss += loss

            model.Backward(y_batch)

            model.update_params(learning_rate)

        avg_loss = running_loss / num_batches
        train_probs = model.Forward(x_train)
        train_acc = compute_accuracy(train_probs, y_train)

        val_probs = model.Forward(x_test)
        val_acc = compute_accuracy(val_probs, y_test)

        #epoch summary

        print(
            f"Epoch:{epoch}/{epochs} - Loss:{avg_loss:0.2f} - Train acc: {train_acc*100:.2f}"
              f"Test acc: {val_acc*100:.2f}"
              )




