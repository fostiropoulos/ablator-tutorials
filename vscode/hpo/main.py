import argparse
from pathlib import Path
import shutil
from typing import Callable, Dict

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from ablator import (
    Literal,
    ModelConfig,
    ModelWrapper,
    ParallelTrainer,
    ParallelConfig,
    configclass,
)
from sklearn.metrics import accuracy_score

# Create a Model config subclass and a run config subclass
# The model config is used to define the model and the
# run config is used to define the training hyperparameters


@configclass
class SimpleConfig(ModelConfig):
    name: Literal["simplenet"]


class SimpleRunConfig(ParallelConfig):
    model_config: SimpleConfig


# Define a simple CNN model using components from PyTorch packages
# And then we wrap up the CNN model in a wrapper class, which defines the loss function,
# forward pass and indicated output formats


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


class MyModel(nn.Module):
    def __init__(self, config: SimpleConfig) -> None:
        super().__init__()
        self.model = SimpleCNN()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels, custom_input=None):
        # custom_input is for demo purposes only, defined in the dataset wrapper
        out = self.model(x)
        loss = self.loss(out, labels)
        if labels is not None:
            loss = self.loss(out, labels)

        out = out.argmax(dim=-1)
        # We need to expand the predictions as they need to be [BS, dim]
        return {"y_pred": out[:, None], "y_true": labels[:, None]}, loss


# Create the training & validation dataloaders from the MNIST dataset.
# Also, data preprocessing is defined here, including normalization and other transformations

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = torchvision.datasets.MNIST(
    root="./datasets", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2
)

testset = torchvision.datasets.MNIST(
    root="./datasets", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2
)


# A evaluation function is definded here for Ablator to evaluate the model and training process.


def my_accuracy(y_true, y_pred):
    return accuracy_score(y_true.flatten(), y_pred.flatten())


# Finally, we wrap up all the defined components in a wrapper class that inherits from ModelWrapper,
# including the model config, the training config, the dataloaders and the evaluation function,
# which is the main interface for Ablator to interact with the model and training process.


class MyModelWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_dataloader_train(self, run_config: SimpleRunConfig):  # type: ignore
        return trainloader

    def make_dataloader_val(self, run_config: SimpleRunConfig):  # type: ignore
        return testloader

    def evaluation_functions(self) -> Dict[str, Callable]:
        return {"accuracy_score": my_accuracy}


# Now we can define the lauching function for Ablator and run the training process.


def run(config: str):
    wrapper = MyModelWrapper(model_class=MyModel)
    run_config = SimpleRunConfig.load(config)
    shutil.rmtree(run_config.experiment_dir, ignore_errors=True)
    ablator = ParallelTrainer(
        wrapper=wrapper,
        run_config=run_config,
    )
    ablator.launch(working_directory=Path.cwd())


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, required=True)
    kwargs = vars(args.parse_args())
    config = run(**kwargs)
