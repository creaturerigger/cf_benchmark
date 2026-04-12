import torch
import torch.nn as nn
import time
from pathlib import Path
import copy


class Trainer:
    def __init__(self, model):
        self.model = model
        self.history = {'epoch': [], 'train_acc': [], 'train_loss': [],
                        'test_acc': [], 'test_loss': [],
                        'model_state_dict': [], 'optimizer_state_dict': []}

    def train(self, epochs=10, criterion=nn.BCELoss(),
              optimizer="adam",
              learning_rate=1e-3,
              train_dataloader: torch.utils.data.DataLoader=None,
              test_dataloader: torch.utils.data.DataLoader=None,
              device='cuda' if torch.cuda.is_available() else 'cpu', save=False,
              model_save_dir: Path=None):

        if save is False and model_save_dir is not None:
            raise ValueError("`model_save_dir` must be None when you set `save` to `False`")
        self.model_save_dir = model_save_dir
        train_steps = len(train_dataloader)
        test_steps = len(test_dataloader)
        self.model.to(device=device)
        if optimizer == "adam" or optimizer is torch.optim.Adam:
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=learning_rate)
        elif isinstance(optimizer, type):
            optimizer = optimizer(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            train_loss = 0.0
            test_loss = 0.0
            correct_train_preds = 0.0
            correct_test_preds = 0.0

            self.model.train()
            for batch_idx, batch in enumerate(train_dataloader):
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device).float().view(-1, 1)
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct_train_preds += (preds == labels).sum().item()

            epoch_train_loss = train_loss / train_steps
            epoch_train_acc = correct_train_preds / len(train_dataloader.dataset)
            self.history['epoch'].append(epoch)
            self.history['train_acc'].append(epoch_train_acc)
            self.history['train_loss'].append(epoch_train_loss)

            self.model.eval()

            for _, test_batch in enumerate(test_dataloader):
                test_features, test_labels = test_batch
                test_features, test_labels = test_features.to(device), \
                                             test_labels.to(device).float().view(-1, 1)
                with torch.no_grad():

                    test_outputs = self.model(test_features)
                    loss = criterion(test_outputs, test_labels)
                    test_preds = (test_outputs > 0.5).float()
                test_loss += loss.item()
                correct_test_preds += (test_preds == test_labels).sum().item()

            epoch_test_loss = test_loss / test_steps
            epoch_test_acc = correct_test_preds / len(test_dataloader.dataset)

            self.history['test_acc'].append(epoch_test_acc)
            self.history['test_loss'].append(epoch_test_loss)
            self.history['optimizer_state_dict'].append(copy.deepcopy(optimizer.state_dict()))
            self.history['model_state_dict'].append(copy.deepcopy(self.model.state_dict()))

        if save:
            self.save_model(self.model_save_dir)

    def save_model(self, root_dir: Path, filename: str = "model.pt"):
        root_dir.mkdir(parents=True, exist_ok=True)
        file_path = root_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
        }, file_path)
