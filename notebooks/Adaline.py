import numpy as np
from sklearn.metrics import accuracy_score
class Adaline:
    def __init__(self):
        import random
        self.random_number = random.random()
        self.weights = None

    def weighting(self, input):
        return np.dot(input, self.weights)

    def activation(self, weighted_input):
        return weighted_input

    def predict(self, inputs):
        # inputs = inputs.to_numpy()
        tmp = np.array(
            [self.activation(self.weighting(np.insert(i, 0, 1))) for i in inputs])
        return np.array([1 if x >= 0 else -1 for x in tmp])

    def fit(self, inputs, outputs, learning_rate, epochs):
        df = inputs
        # inputs = inputs.to_numpy()
        # outputs = outputs.to_numpy()
        inputs = np.array([np.insert(i, 0, 1) for i in inputs])
        self.weights = [self.random_number for _ in range(len(inputs[0]))]

        for _ in range(epochs):

            counter = 0
            tmp = 0
            for i in inputs:
                wei = self.weighting(i)
                act = self.activation(wei)

                tmp += (outputs[counter] - act)*i
                counter += 1

            deltaWi = learning_rate*tmp
            self.weights = self.weights + deltaWi

            print(accuracy_score(self.predict(df), outputs))

