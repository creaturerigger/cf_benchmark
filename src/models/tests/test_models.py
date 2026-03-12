import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.pytorch_classifier import PYTModel
from src.models.trainer import Trainer

IN_FEATURES = 16
N_SAMPLES = 100
BATCH_SIZE = 32

BASE_CFG = {
    "model": {
        "layers": [
            {"out_features": 20, "activation": "relu", "batch_norm": False, "dropout": 0.0}
        ],
        "output_activation": "sigmoid"
    }
}


@pytest.fixture
def model():
    return PYTModel(in_features=IN_FEATURES, cfg=BASE_CFG)


@pytest.fixture
def dataloaders():
    X = torch.randn(N_SAMPLES, IN_FEATURES)
    y = torch.randint(0, 2, (N_SAMPLES,))
    train_ds = TensorDataset(X[:80], y[:80])
    test_ds = TensorDataset(X[80:], y[80:])
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE),
        DataLoader(test_ds, batch_size=BATCH_SIZE),
    )


# --- PYTModel tests ---

def test_model_output_shape(model):
    x = torch.randn(BATCH_SIZE, IN_FEATURES)
    out = model(x)
    assert out.shape == (BATCH_SIZE, 1)


def test_model_output_range(model):
    x = torch.randn(200, IN_FEATURES)
    out = model(x)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_model_architecture():
    cfg = {
        "model": {
            "layers": [
                {"out_features": 32, "activation": "relu", "batch_norm": True, "dropout": 0.1},
                {"out_features": 16, "activation": "tanh", "batch_norm": False, "dropout": 0.0},
            ],
            "output_activation": "sigmoid"
        }
    }
    m = PYTModel(in_features=IN_FEATURES, cfg=cfg)
    out = m(torch.randn(8, IN_FEATURES))
    assert out.shape == (8, 1)


def test_model_no_output_activation():
    cfg = {"model": {"layers": [{"out_features": 20, "activation": "relu"}], "output_activation": None}}
    m = PYTModel(in_features=IN_FEATURES, cfg=cfg)
    out = m(torch.randn(8, IN_FEATURES))
    assert out.shape == (8, 1)


# --- Trainer tests ---

def test_trainer_history_populated(model, dataloaders):
    train_dl, test_dl = dataloaders
    trainer = Trainer(model)
    trainer.train(epochs=2, train_dataloader=train_dl, test_dataloader=test_dl, device='cpu')
    assert len(trainer.history['epoch']) == 2
    assert len(trainer.history['train_acc']) == 2
    assert len(trainer.history['test_acc']) == 2
    assert len(trainer.history['train_loss']) == 2
    assert len(trainer.history['test_loss']) == 2


def test_trainer_accuracy_range(model, dataloaders):
    train_dl, test_dl = dataloaders
    trainer = Trainer(model)
    trainer.train(epochs=3, train_dataloader=train_dl, test_dataloader=test_dl, device='cpu')
    for acc in trainer.history['train_acc'] + trainer.history['test_acc']:
        assert 0.0 <= acc <= 1.0


def test_trainer_save(model, dataloaders, tmp_path):
    train_dl, test_dl = dataloaders
    trainer = Trainer(model)
    trainer.train(epochs=1, train_dataloader=train_dl, test_dataloader=test_dl,
                  device='cpu', save=True, model_save_dir=tmp_path)
    assert len(list(tmp_path.glob("*.pt"))) == 1


def test_trainer_save_requires_dir(model, dataloaders, tmp_path):
    train_dl, test_dl = dataloaders
    trainer = Trainer(model)
    with pytest.raises(ValueError):
        trainer.train(epochs=1, train_dataloader=train_dl, test_dataloader=test_dl,
                      device='cpu', save=False, model_save_dir=tmp_path)



@pytest.fixture
def load_model_params():
    with open('configs/model/pytorch_classifier.yaml') as f:
        return yaml.safe_load(f)
