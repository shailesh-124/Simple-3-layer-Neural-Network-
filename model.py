import numpy as np
#Making a three layer model
class SimpleNeuralNetwork:
    def __init__(self, input_size=784, output_size=10, hidden_size=64, seed=42):
        rng = np.random.default_rng(seed)
        
        self.w1 = rng.normal(0.0, 0.1, size=(input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)

        self.w2 = rng.normal(0.0, 0.1, size=(hidden_size, output_size))
        self.b2 = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(x, 0)
    
    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    def Forward(self, X):
        """
        Forward propagation passes the image grayscale values through the network layers
        X size = (batch_size, 784)
        """
        #HIDDEN LAYER
        self.z1 = (X @ self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        #OUTPUT LAYER
        self.z2 = (self.a1 @ self.w2) + self.b2
        self.probabilities = self.softmax(self.z2)
        self.X = X
        return self.probabilities

    def Backward(self, y_true_one_hot):
        """
        computes gradients for weights and biases using backpropagation
        y_true_one_hot size = (batch_size, 10)
        """
        batch_size = self.X.shape[0]
        #error
        dz2 = self.probabilities - y_true_one_hot

        #output layer
        self.dw2 = (self.a1.T @ dz2 ) / batch_size
        self.db2 = np.sum(dz2, axis=0) / batch_size

        #hidden layer
        da1 = (dz2 @ self.w2.T) 
        dz1 = da1 *(self.z1 > 0)

        self.dw1 = (self.X.T @ dz1) / batch_size
        self.db1 = np.sum(dz1, axis=0) / batch_size

    def update_params(self, learning_rate = 0.1):
        self.w1 -= learning_rate*self.dw1
        self.b1 -= learning_rate*self.db1
        self.w2 -= learning_rate*self.db2
        self.b2 -= learning_rate*self.db2

    def save_weights(self, filepath="model_weights.npz"):
        """saves weights and biases to a compressed npz file"""
        np.savez(filepath, w1=self.w1, w2=self.w2, b1=self.b1, b2=self.b2)
        print(f"Model parameters saved to {filepath}")

    def load_weights(self, filepath="model_weights.npz"):
        "loads the weights from the .npz file"

        weights = np.load(filepath)
        self.w1 = weights["w1"]
        self.w2 = weights["w2"]
        self.b1 = weights["b1"]
        self.b2 = weights["b2"]


