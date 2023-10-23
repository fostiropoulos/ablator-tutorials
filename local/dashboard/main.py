from pathlib import Path
import shutil
from torch import nn
import torch
from ablator import (
    ModelConfig,
    ModelWrapper,
    OptimizerConfig,
    TrainConfig,
    configclass,
    Literal,
    ParallelTrainer,
    SearchSpace,
    ParallelConfig,
    RemoteConfig,
)

remote_config = RemoteConfig(
    s3={
        "provider": "AWS",
        "region": "us-east-2",
        "access_key_id": "AKIAU5WVS5VGKFYYCPMH",
        "secret_access_key": "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",
    },
    remote_path=Path("ablator-mock"),  # s3://some-bucket
)


@configclass
class TrainConfig(TrainConfig):
    dataset: str = "random"
    dataset_size: int


@configclass
class ModelConfig(ModelConfig):
    layer: Literal["layer_a", "layer_b"] = "layer_a"


@configclass
class ParallelConfig(ParallelConfig):
    model_config: ModelConfig
    train_config: TrainConfig


config = ParallelConfig(
    experiment_dir=Path("/tmp/ablator-exp"),
    train_config=TrainConfig(
        batch_size=128,
        epochs=5,
        dataset_size=100,
        optimizer_config=OptimizerConfig(name="sgd", arguments={"lr": 0.1}),
        scheduler_config=None,
    ),
    model_config=ModelConfig(),
    device="cpu",
    search_space={
        "model_config.layer": SearchSpace(categorical_values=["layer_a", "layer_b"])
    },
    total_trials=10,
    remote_config=remote_config,
)


class SimpleModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.layer == "layer_a":
            self.param = nn.Parameter(torch.ones(50, 1))
        else:
            self.param = nn.Parameter(torch.randn(200, 1))

    def forward(self, x: torch.Tensor):
        x = self.param * x
        return {"preds": x}, x.sum().abs()


class SimpleWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: ParallelConfig):
        dl = [torch.rand(100) for i in range(run_config.train_config.dataset_size)]
        return dl

    def make_dataloader_val(self, run_config: ParallelConfig):
        dl = [torch.rand(100) for i in range(run_config.train_config.dataset_size)]
        return dl


if __name__ == "__main__":
    mywrapper = SimpleWrapper(SimpleModel)
    shutil.rmtree(config.experiment_dir, ignore_errors=True)
    with ParallelTrainer(mywrapper, config) as runner:
        runner.launch(".")
