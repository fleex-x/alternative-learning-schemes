import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from typing import List
from common import model_size

class TrainingStats:
    loss: List[float]
    grad_norm: List[float]
    train_accuracy: List[float]

    def __init__(self) -> None:
        self.loss = []
        self.grad_norm = []
        self.train_accuracy = []

    def update_stats(self, loss: float, grad_norm: float, train_accuracy: float):
        self.loss.append(loss)
        self.grad_norm.append(grad_norm)
        self.train_accuracy.append(train_accuracy)
    
    def collect_stats(self, model: nn.Module, dataloader: DataLoader, device: torch.device):
        loss_function = nn.CrossEntropyLoss(reduction='sum')
        to_train = list(filter(lambda p: p.requires_grad , model.parameters()))
        gradients = [torch.zeros_like(to_train[i].data) for i in range(len(to_train))]
        loss = 0
        cnt_data = 0
        cnt_true_predictions = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), torch.flatten(targets).to(device).long()
            outputs = model.forward(inputs)
            current_loss: torch.Tensor = loss_function(outputs, targets)
            current_loss.backward()

            with torch.no_grad():
                predicted = outputs.max(dim=1)[1]
                cnt_true_predictions += torch.sum(predicted == targets).item()
                for i, p in enumerate(to_train):
                    gradients[i] += p.grad
                loss += current_loss.item()

            cnt_data += targets.size(0)
        grad_norm = 0
        accuracy = cnt_true_predictions / cnt_data
        loss /= cnt_data
        with torch.no_grad():
            for g in gradients:
                g /= cnt_data
                grad_norm += g.pow(2.0).sum().item()
            grad_norm = grad_norm ** 0.5
        
        self.update_stats(loss, grad_norm, accuracy)



def lbfgs_train(model: nn.Module,
                n: int,
                dataloader: DataLoader,
                device: torch.device,
                max_eval: int = 50,
                history_size: int = 50) -> TrainingStats:
    print(f"LBFGS training, params {model_size(model)}\n")
    model.train()
    to_train = list(filter(lambda p: p.requires_grad , model.parameters()))
    optimizer = optim.LBFGS(params=to_train, max_eval=max_eval, history_size=history_size, 
                            line_search_fn='strong_wolfe', lr=0.3)
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    epoch_num = [0]
    stats = TrainingStats()

    def closure() -> torch.Tensor:
        gradients = [torch.zeros_like(to_train[i].data) for i in range(len(to_train))]
        optimizer.zero_grad()
        loss = 0
        cnt_data = 0
        cnt_true_predictions = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), torch.flatten(targets).to(device).long()
            outputs = model.forward([inputs, n])
            current_loss: torch.Tensor = loss_function(outputs, targets)
            current_loss.backward()

            with torch.no_grad():
                predicted = outputs.max(dim=1)[1]
                cnt_true_predictions += torch.sum(predicted == targets).item()
                for i, p in enumerate(to_train):
                    gradients[i] += p.grad
                loss += current_loss

            cnt_data += targets.size(0)

        grad_norm = 0
        with torch.no_grad():
            for i, p in enumerate(to_train):
                p.grad = gradients[i] / cnt_data
                grad_norm += p.grad.pow(2.0).sum().item()
            loss /= cnt_data
            grad_norm = grad_norm ** 0.5

        stats.update_stats(loss.item(), grad_norm, cnt_true_predictions/cnt_data)

        print(f"Epoch #{epoch_num[0]} loss is {loss.item()}")
        epoch_num[0] += 1
        return loss
    optimizer.step(closure)
    return stats
        
def adam_train(model: nn.Module,
               n: int,  
               dataloader: DataLoader,
               device: torch.device,
               max_eval: int = 50) -> TrainingStats:
    print(f"ADAM training, params {model_size(model)}\n")
    model.train()
    to_train = list(filter(lambda p: p.requires_grad , model.parameters()))
    optimizer = optim.Adam(params=to_train)
    stats = TrainingStats()

    loss_function = nn.CrossEntropyLoss(reduction='mean')
    for epoch in range(max_eval):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), torch.flatten(targets).long().to(device)
            optimizer.zero_grad()
            loss = loss_function(model.forward([inputs, n]), targets)
            loss.backward()
            optimizer.step()

        stats.collect_stats(model, dataloader, device)
        print(f"Epoch #{epoch} loss is {stats.loss[-1]}")
    return stats