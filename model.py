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

        return np.zeros_like(self.x) # we wont use so just return zeros
        
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
        # dy: [batch_size, ..., output_size]
        dw = np.sum(self.x.transpose(0, 2, 1) @ dy, axis=0) 
        db = dy.sum(axis=(0, 1), keepdims=True).reshape(1, -1)
        dx = dy @ self.w.T
        self.w -= dw * lr
        self.b -= db * lr
        
        return dx

class ScaledDotProduct():
    def __init__(self):
        pass
    def forward(self, Q, K, V):
        self.Q, self.K, self.V = Q, K, V
        d_k = K.shape[-1]

        self.scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_k)  # [batch_size, seq_length, seq_length]
        self.A = self.softmax(self.scores) # [batch_size, seq_length, seq_length]
        self.y = self.A @ V # [batch_size, seq_length, hidden_size]
        return self.y

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def backward(self, dy):
        # [batch_size, seq_length, hidden_size]
        Q, K, V, A = self.Q, self.K, self.V, self.A
        d_k = K.shape[-1]
        B = dy.shape[0]

        # dY = A @ V
        dA = dy @ V.transpose(0, 2, 1) # [batch_size, seq_length, seq_length]
        dV = A.transpose(0, 2, 1) @ dy # [batch_size, seq_length, hidden_size]

        # softmax grad
        # dS = A * (dA - (A * dA).sum(-1, keepdims=True))
        tmp = (A * dA).sum(axis=-1, keepdims=True)
        dS = A * (dA - tmp) # [batch_size, seq_length, seq_length]

        # S = (Q @ K^T) / sqrt(d_k)
        dQ = dS @ K / np.sqrt(d_k) # [batch_size, seq_length, hidden_size // n_heads]
        dK = dS.transpose(0, 2, 1) @ Q / np.sqrt(d_k) # [batch_size, seq_length, hidden_size // n_heads]

        return dQ, dK, dV

class SelfAttention:
    def __init__(self, hidden_size, d_k):
        self.Wq = Linear(hidden_size, d_k)
        self.Wk = Linear(hidden_size, d_k)
        self.Wv = Linear(hidden_size, d_k)
        self.attn = ScaledDotProduct()

    def forward(self, x):
        self.x = x  # store for backward
        Q = self.Wq.forward(x)
        K = self.Wk.forward(x)
        V = self.Wv.forward(x)
        out = self.attn.forward(Q, K, V)
        return out

    def backward(self, dy, lr):
        # from attention backward
        dQ, dK, dV = self.attn.backward(dy)

        # propagate into Q,K,V projections
        dx_q = self.Wq.backward(dQ, lr)
        dx_k = self.Wk.backward(dK, lr)
        dx_v = self.Wv.backward(dV, lr)

        # total input grad
        dx = dx_q + dx_k + dx_v
        return dx

        
        
