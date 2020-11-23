import numpy as np
import random
from data_loader import DataLoader

def sigmoid(inX):
    return 1.0 / (1.0 + np.exp(-inX))

def relu(inX):
    return np.maximum(inX, 0)

class NNRegressor:
    def __init__(self, learning_rate=0.002, hidden_size=48, max_epochs=300, schedule_point=(), gamma=0.1, batch_size=64):
        self.learning_rate = learning_rate
        self.max_epochs= max_epochs
        self.batch_size=batch_size
        self.hidden_layer = None
        self.output_layer = None
        self.hidden_size = hidden_size
        self.schedule_point = schedule_point
        self.gamma = gamma
        
        
    
    def fit(self, X:np.array, y:np.array):
        self.X_train = X
        self.y_train = y
        data_loader = DataLoader(X=X, y=y, batch_size=self.batch_size)
        item_length = data_loader.length + 1
        self.hidden_layer = np.random.normal(size=(self.hidden_size, item_length)) # (hidden_size, input_length +1)
        self.output_layer = np.random.normal(size=(self.hidden_size, 1)) # (hidden_size, 1)
        self.output_bias = random.random()
        
        
        for epoch in range(self.max_epochs):
            total_error = 0
            if epoch in self.schedule_point:
                print("Decrease lr from {} to {}".format(self.learning_rate, self.gamma * self.learning_rate))
                self.learning_rate *= self.gamma
            for batch in range(len(data_loader)):
                item_X, item_y = data_loader[batch]
                item_X = np.c_[item_X, np.ones(shape=(self.batch_size,1))]
                # forward
                hidden_output = np.dot(item_X, self.hidden_layer.T) # (batch_size, hidden_size)
                output_input = sigmoid(hidden_output)# (batch_size, hidden_size)
                output = np.dot(output_input, self.output_layer).T + self.output_bias# (1, batch_size)
                np.reshape(output, (self.batch_size,)) #(1, batch_size)
                mse_error = np.sum((item_y - output)**2) / self.batch_size
                total_error += mse_error
                
                # backward
                # 输出层偏置
                gradient_output_bias = 2 / self.batch_size
                gradient_output_bias *= np.sum(output-item_y)
                self.output_bias -= self.learning_rate * gradient_output_bias
                
                # 输出层权值
                gradient_output_weight = 2 / self.batch_size
                gradient_output_weight *= np.dot((output-item_y),output_input).T #(hidden_size, 1) 
                self.output_layer -= self.learning_rate * gradient_output_weight
                
                # 隐藏层权值
                gradient_hidden_weight = 2 / self.batch_size
                gradient_hidden_weight *= np.dot(self.output_layer, output-item_y) #(hidden_size, batch_size)
                derivative_activate = (1-output_input) * output_input #(batch_size, hidden_size)
                gradient_hidden_weight *= derivative_activate.T #(hidden_size, batch_size)
                gradient_hidden_weight = np.dot(gradient_hidden_weight, item_X) #(hidden, input_length + 1)
                self.hidden_layer -= self.learning_rate * gradient_hidden_weight
                
            print("[Epoch {:>3d}/{}]: MSE Training Loss: {:.8f}.".format(epoch + 1, self.max_epochs, total_error / len(data_loader)))
                
    def predict(self, X: np.array):
        inX = np.c_[X, np.ones(shape=(len(X), 1))]
        hidden_output = np.dot(inX, self.hidden_layer.T)
        hidden_output = sigmoid(hidden_output)
        output = np.dot(hidden_output, self.output_layer).T + self.output_bias
        return output