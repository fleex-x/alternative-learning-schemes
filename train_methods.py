import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from typing import List, Any, Optional, Tuple
from dataclasses import dataclass
from sparse_lbfgs import SparseLBFGS

@dataclass
class SamplesStats:
    loss: torch.Tensor
    gradients: List[torch.Tensor]
    grad_norm: float
    accuracy: float

def loss_grad_acc_over_samples(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device) -> SamplesStats:
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
            loss += current_loss

        cnt_data += targets.size(0)

    grad_norm = 0
    accuracy = cnt_true_predictions / cnt_data
    loss /= cnt_data
    with torch.no_grad():
        for g in gradients:
            g /= cnt_data
            grad_norm += g.square().sum().item()
        grad_norm = grad_norm ** 0.5
    return SamplesStats(loss, gradients, grad_norm, accuracy) 

def get_dist(current_points: List[torch.Tensor], prev_points: List[torch.Tensor]) -> float:
    res = 0
    for i in range(len(current_points)):
        res += (current_points[i] - prev_points[i]).square().sum().item()
    return res

class TrainingStats:
    loss: List[float]
    grad_norm: List[float]
    train_accuracy: List[float]
    step_size: List[float]
    prev_points: Any

    def __init__(self) -> None:
        self.loss = []
        self.grad_norm = []
        self.train_accuracy = []
        self.step_size = []
        self.prev_points = None

    def update_stats(self, 
                     loss: float, 
                     grad_norm: float, 
                     train_accuracy: float,
                     step_size: Optional[float] = None):
        self.loss.append(loss)
        self.grad_norm.append(grad_norm)
        self.train_accuracy.append(train_accuracy)
        if step_size is not None:
            self.step_size.append(step_size)
    
    def collect_stats(self, model: nn.Module, dataloader: DataLoader, device: torch.device):
        samples_stats = loss_grad_acc_over_samples(model, dataloader, device)
        to_train = list(filter(lambda p: p.requires_grad , model.parameters()))
        if self.prev_points is not None:
            self.update_stats(samples_stats.loss.item(), 
                              samples_stats.grad_norm, 
                              samples_stats.accuracy, 
                              get_dist([p.data for p in to_train], self.prev_points))
        else:
            self.update_stats(samples_stats.loss.item(), 
                              samples_stats.grad_norm, 
                              samples_stats.accuracy)
        
        self.prev_points = [p.data.clone() for p in to_train]


def lbfgs_train(model: nn.Module,
                dataloader: DataLoader,
                device: torch.device,
                max_eval: int = 50,
                history_size: int = 50) -> TrainingStats:
    print("\n\nL-BFGS train\n\n")
    model.train()
    to_train = list(filter(lambda p: p.requires_grad , model.parameters()))
    optimizer = optim.LBFGS(params=to_train, max_eval=max_eval, history_size=history_size, 
                            line_search_fn='strong_wolfe', lr=0.3)
    epoch_num = [0]
    stats = TrainingStats()
    prev_points = [None]

    def closure() -> torch.Tensor:
        samples_stats = loss_grad_acc_over_samples(model, dataloader, device)

        step_size = None
        with torch.no_grad():
            if prev_points[0] is not None:
                step_size = get_dist([p.data for p in to_train], prev_points[0])

            prev_points[0] = [0] * len(to_train)
            for i, p in enumerate(to_train):
                p.grad = samples_stats.gradients[i]
                prev_points[0][i] = p.data.clone()

        if step_size is None:
            stats.update_stats(samples_stats.loss.item(), 
                               samples_stats.grad_norm, 
                               samples_stats.accuracy)
        else:
            stats.update_stats(samples_stats.loss.item(), 
                               samples_stats.grad_norm, 
                               samples_stats.accuracy, 
                               step_size)

        print(f"Epoch #{epoch_num[0]} loss is {samples_stats.loss.item()}")
        epoch_num[0] += 1
        return samples_stats.loss

    optimizer.step(closure)
    return stats
        
def adam_train(model: nn.Module,
               dataloader: DataLoader,
               device: torch.device,
               max_eval: int = 50) -> TrainingStats:
    print("\n\nAdam train\n\n")
    model.train()
    to_train = list(filter(lambda p: p.requires_grad , model.parameters()))
    optimizer = optim.Adam(params=to_train)
    stats = TrainingStats()

    loss_function = nn.CrossEntropyLoss(reduction='mean')
    for epoch in range(max_eval):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), torch.flatten(targets).long().to(device)
            optimizer.zero_grad()
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()

        stats.collect_stats(model, dataloader, device)
        print(f"Epoch #{epoch} loss is {stats.loss[-1]}")
    return stats

def combined_train(model: nn.Module,
                   dataloader: DataLoader,
                   device: torch.device,
                   max_eval: int = 50) -> Tuple[TrainingStats, TrainingStats]:
    print("\n\nCombined train\n\n")

    model.train()
    to_train = list(filter(lambda p: p.requires_grad , model.parameters()))
    optimizer = optim.Adam(params=to_train)
    stats = TrainingStats()

    loss_function = nn.CrossEntropyLoss(reduction='mean')
    for epoch in range(max_eval):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), torch.flatten(targets).long().to(device)
            optimizer.zero_grad()
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()

        stats.collect_stats(model, dataloader, device)
        print(f"Epoch #{epoch} loss is {stats.loss[-1]}")
        if stats.train_accuracy[-1] >= 0.5:
            return stats, lbfgs_train(model, dataloader, device, max(5, max_eval - epoch - 1))
    return stats, lbfgs_train(model, dataloader, device, 5)




def sparse_lbfgs_train(model: nn.Module,
                dataloader: DataLoader,
                device: torch.device,
                max_eval: int = 50,
                history_size: int = 50) -> TrainingStats:
    print("\n\nL-BFGS train\n\n")
    model.train()
    to_train = list(filter(lambda p: p.requires_grad , model.parameters()))
    epoch_num = [0]
    stats = TrainingStats()
    prev_points = [None]

    def closure() -> torch.Tensor:
        samples_stats = loss_grad_acc_over_samples(model, dataloader, device)

        step_size = None
        with torch.no_grad():
            if prev_points[0] is not None:
                step_size = get_dist([p.data for p in to_train], prev_points[0])

            prev_points[0] = [0] * len(to_train)
            for i, p in enumerate(to_train):
                p.grad = samples_stats.gradients[i]
                prev_points[0][i] = p.data.clone()

        if step_size is None:
            stats.update_stats(samples_stats.loss.item(), 
                               samples_stats.grad_norm, 
                               samples_stats.accuracy)
        else:
            stats.update_stats(samples_stats.loss.item(), 
                               samples_stats.grad_norm, 
                               samples_stats.accuracy, 
                               step_size)

        print(f"Epoch #{epoch_num[0]} loss is {samples_stats.loss.item()}")
        epoch_num[0] += 1
        return samples_stats.loss

    optimizer = SparseLBFGS(param_groups=[to_train], func=None, func_grad=closure, history_size=history_size)
    for _ in range(max_eval):
        optimizer.optimization_step()
        if optimizer.finished():
            break

    optimizer.step(closure)
    return stats