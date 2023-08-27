# Train a simple CNN model with Ablator
This example shows how to train a LeNet-5 model with Ablator with the MNIST dataset on VSCode.

## Prerequisites

- **OS**: Linux or macOS
- **Python**: 3.10 or higher
- **IDE**: VSCode or Pycharm



Initialize the virtual environment and install the dependencies.

```bash
python3 -m venv env
source env/bin/activate
pip install ablator
```

Last, you have to create a temporary directory for the experiment results. We choose `experiment_dir` for this example.

```bash
mkdir -p experiment_dir
```

## Run the experiment

To run a demo experiment, you can directly use the demo codes `./main.py`, along with the configuration file `./config.yaml`.

```bash
python3 ./main.py --config config
```

If you can see the experiment process printed on the terminal, then you have successfully run the demo experiment.

## Check the results

The experiment results are cached in the temporary directory `/tmp/dir`. Each experiment is allocated with a unique ID. Please check the results by running the following command.

```bash
cd experiment_dir/<experiment_id>
cat results.json
```

You should see the basic metrics of the model after each epoch. Also, you can retrieve the configurations by checking the file `./config.yaml`. The trained model is cached in the directory `./checkpoints/`.

---

## Customize model structures

Since Ablatror is designed to help researchers focus more on the model itself and remove most boilerplate code, you can easily change your model structures with less effort. The initial model structure is defined in the file `./main.py`.

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x
```

The initial model is a simple CNN model with two convolutional layers and two fully connected layers. You can easily change the model structure by modifying the code above. For example, you can add more convolutional layers by adding the following code.

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        ...
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        ...

    def forward(self, x):
        ...
        x = self.pool3(self.relu5(self.conv3(x)))
        ...
```

**Please note that**: to input the customized model into Ablator, you have to wrap up the model with a Ablator `ModelWrapper` class, where you should define the loss function, evaluation functions and data loaders.

## Customize training parameters
All training-related configurations are defined in the file `./config.yaml`. You can easily change the training parameters by modifying the file. For example, you can change the batch size by modifying the following code.

```bash
train_config:
  dataset: mnist
  optimizer_config:
    name: sgd
    arguments:
      lr: 0.001
      momentum: 0.9
  batch_size: 64
  epochs: 20
  scheduler_config: null
  rand_weights_init: true
```

- `epoch`: Number of training epochs.
- `batch_size`: Size of each training batch.
- `optimizer_config`: configurations of the optimizer. You can change the name of the optimizer and also the optimizer arguments.

Please check ABLATOR [documentation](https://ablator.org) for more details about the configurations.

## Switch datasets

In this experiment demo, we use the datasets from PyTorch packages. You can easily change to other datasets by modifying the code in `./main.py`. For example, you can change to **CIFAR-10** dataset by adding the following code.

```python
trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
```

**Please note that:** if you change the dataset, the data pre-processing and model structure may also need to be changed accordingly, such as the image normalizations and input/output dimensions.

Also, to input the datasets into Ablator, you should wrap up the datasets with a dataloader and return it in the `ModelWrapper` class, with the method `make_dataloader_train` or `make_dataloader_test`, for training and testing respectively.

---

## Conclusions

In this example, we show how to train a simple CNN model with Ablator under Linux/MacOS environments. The Ablator framework removes most of the boil-plate codes and allows us to only implement the core codes of a model and experiments. You can also easily modify the important parameters of both the model and the training process. We hope this example can help you to get started with Ablator.



