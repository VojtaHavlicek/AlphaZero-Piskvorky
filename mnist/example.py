import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from tqdm import tqdm
import numpy as np

# --- MULTI-LAYER PERCEPTRON (MLP) ---
class MLP(nn.Module): 
    def __init__( 
            self, num_layers: int, input_dim:int, hidden_dim:int, output_dim:int
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [ 
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """ Forward pass through the MLP """
        for layer in self.layers[:-1]:
            x = mx.maximum(0, layer(x))  # ReLU activation 
        return self.layers[-1](x)
    
def loss_fn(model, X, y): 
    return mx.mean(nn.losses.cross_entropy(model(X), y))
    
def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)
    

# --- Model Parameters ---
num_layers = 2
hidden_dim = 32
num_classes = 10  # For MNIST, we have 10 classes (digits 0-9)
batch_size = 256
num_epochs = 20
learning_rate = 0.001 # 0.1 is good for SGD, but high for Adam (Yea!)

# --- Dataset --- 
import mnist 

train_images, train_labels, test_images, test_labels = map(mx.array, mnist.mnist())

def batch_iterate(batch_size, X, y): 
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s: s + batch_size]
        yield X[ids], y[ids]


model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)
mx.eval(model.parameters())

# Loss and gradient with respect to the model trainable parameters
loss_and_grad_fn = nn.value_and_grad(model, fn=loss_fn)

# Optimizer 
optimizer = optim.Adam(learning_rate=learning_rate)

for e in tqdm(range(num_epochs)):
    for X, y in batch_iterate(batch_size, train_images, train_labels):
        loss, grads = loss_and_grad_fn(model, X, y)

        optimizer.update(model, grads)

        mx.eval(model.parameters(), optimizer.state)

    accuracy = eval_fn(model, test_images, test_labels)
    print(f"Epoch {e + 1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

