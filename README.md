# tnn
Matrix-Mimetic Tensor Neural Networks 

## Installation

Using `pip`
```console
python -m pip install git+https://<personal_access_token>@github.com/elizabethnewman/tnn.git
```

Using `github`
```console
git clone https://github.com/elizabethnewman/tnn.git
```

## Quick Start

Let's train a network on to classify MNIST data.  First, we load the data
```python
import torch
from torchvision import datasets, transforms

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
    
train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)

# make the dataset smaller to test if code runs
n_train = 1000
n_test = 100

dataset1.data = dataset1.data[:n_train]
dataset2.data = dataset2.data[:n_test]

train_loader = torch.utils.data.DataLoader(dataset1, batch_size=32)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=32)
```

Next, we will create a standard fully-connected network and train it!
```python
from tnn.layers import View
from tnn.networks import FullyConnected
from tnn.training.batch_train import train
from tnn.utils import seed_everything, number_network_weights

# seed for reproducibility
seed_everything(1234)

# form network
net = torch.nn.Sequential(View((-1, 784)), 
                          FullyConnected([784, 20, 10], activation=torch.nn.Tanh())
                          )
                          
# choose loss function
loss = torch.nn.CrossEntropyLoss()

# choose optimizer
optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-2)
                    
# train!
results = train(net, loss, optimizer, train_loader, test_loader, max_epochs=10, verbose=True)
```

