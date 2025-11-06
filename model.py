import numpy as np

class Embedding():
    def __init__(self, vocab_size, n_embd):
        # input: [batch_size, seq_length]
        # ouput: [batch_size, seq_length, n_embd]
    
        limit = 1 / np.sqrt(n_embd)
        self.w_e = np.random.uniform(-limit, limit, (vocab_size, n_embd))
    
    def forward(self, x):
        self.x = x  # store input for backward pass
        y = self.w_e[x]
        return y

    def backward(self, dy, lr):
        # numpyish way to quickly calculate the gradient
        # dy.shape = w_e.shape : [vocab_size, n_embd]

        self.dw_e = np.zeros_like(self.w_e)
        np.add.at(self.dw_e, self.x.reshape(-1), dy.reshape(-1, dy.shape[-1]))
        self.w_e -= self.dw_e * lr
        
class Linear():
    def __init__(self, hidden_size, output_size):
        # xavier init
        limit = np.sqrt(6 / (hidden_size + output_size))
        self.w = np.random.uniform(-limit, limit, (hidden_size, output_size))
        self.b = np.zeros((1, output_size))

    def forward(self, x):
        # x: [batch_size, ..., hidden_size]
        # y: [batch_size, ..., output_size] 
        self.x = x
        y = x @ self.w + self.b

        return y

    def backward(self, dy, lr):
        # dy: [batch_size, output_size]

        dw = self.x.T @ dy
        db = dy.sum(axis=0, keepdims=True) 
        dx = dy @ self.w.T
        self.w -= dw * lr
        self.b -= db * lr
        
        return dx
