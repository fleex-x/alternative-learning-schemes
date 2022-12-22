from typing import List, Callable, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
from enum import Enum


class ViewAsFlatTensor:
    tensors: List[torch.Tensor]
    sum_size: int

    def __init__(self, tensors: List[torch.Tensor]) -> None:
        self.tensors = []
        self.sum_size = 0
        for t in tensors:
            self.tensors.append(torch.flatten(t))
            self.sum_size += self.tensors[-1].size()[0]


    def ith(self, key: int) -> torch.Tensor:
        assert key < self.sum_size
        i = 0
        while self.tensors[i].size()[0] >= key:
            key -= self.tensors[i].size()[0]
            i += 1
        return self.tensors[i][key]

    def size(self) -> int:
        return self.sum_size

    def norm(self) -> torch.Tensor:
        res = torch.zeros(1)
        for t in self.tensors:
            res += t.square().sum()
        return res.sqrt()

    def mul_scalar(self, alpha) -> 'ViewAsFlatTensor':
        for t in self.tensors:
            t *= alpha
        return self

    def mul_tensor(self, view: 'ViewAsFlatTensor') -> 'ViewAsFlatTensor':
        for i, t in enumerate(self.tensors):
            t *= view.tensors[i]
        return self

    def add_tensor(self, other: 'ViewAsFlatTensor', mul = 1) -> 'ViewAsFlatTensor':
        for i in range(len(self.tensors)):
            self.tensors[i] += other.tensors[i] * mul
        return self

    def clone(self) -> 'ViewAsFlatTensor':
        return ViewAsFlatTensor([t.clone() for t in self.tensors])

    def zeros_like(self) -> 'ViewAsFlatTensor':
        return ViewAsFlatTensor([torch.zeros_like(t) for t in self.tensors])
        


def concat(views: List[ViewAsFlatTensor]) -> ViewAsFlatTensor:
    result = []
    for v in views:
        result.extend(v.tensors)
        result += v.sum_size
    return ViewAsFlatTensor(result)
        

def dot_same_shape(a: ViewAsFlatTensor, b: ViewAsFlatTensor) -> torch.Tensor:
    res = 0
    for i in range(len(a.tensors)):
        res += torch.tensordot(a.tensors[i], b.tensors[i])
    return res


class LBFGSHistory:
    __points_delta: List[ViewAsFlatTensor] # delta for each param group
    __grads_delta: List[ViewAsFlatTensor] # delta for each param group
    __p: List[torch.Tensor] # 1/dot(__points_delta, __grads_delta)

    history_size: int

    def __init__(self, history_size: int) -> None:
        self.history_size = history_size
        self.__points_delta = []
        self.__grads_delta = []

    def update_history(self,
                       point_delta: ViewAsFlatTensor, 
                       grad_delta: ViewAsFlatTensor):
        self.__points_delta.append(point_delta)
        self.__grads_delta.append(grad_delta)
        self.__p.append(1./dot_same_shape(point_delta, grad_delta))
        if len(point_delta) > self.history_size:
            self.__points_delta.pop(0)
            self.__grads_delta.pop(0)
            self.__p.remove(0)

    def compute_descent_direction(self, grad: ViewAsFlatTensor) -> ViewAsFlatTensor:
        res = grad.clone()
        m = len(self.__grads_delta)
        if m == 0:
            return res
        a = []
        for i in range(m):
            a.append(self.__p[i] * dot_same_shape(self.__points_delta[i], res))
            res.add_tensor(self.__grads_delta[i], mul=-a[i])
        res.mul_scalar(
            dot_same_shape(self.__points_delta[-1], self.__grads_delta[-1]) / 
            dot_same_shape(self.__grads_delta[-1], self.__grads_delta[-1])
        )
        for i in range(m - 1, -1, -1):
            b = (self.__p[i] * dot_same_shape(self.__grads_delta[i], res))
            res.add_tensor(self.__grads_delta[i], mul=(a[i] - b))



def wolfe_search(
        point: ViewAsFlatTensor,
        start_loss: torch.Tensor, 
        start_grad: ViewAsFlatTensor, 
        direction: ViewAsFlatTensor, 
        lr: float,
        func: Callable[[ViewAsFlatTensor], Tuple[torch.Tensor, ViewAsFlatTensor]]
    ) -> torch.Tensor:
    c1 = 1e-9
    c2 = 1e-4
    step = lr
    while True:    
        cur_loss, cur_grad = func(point.сlone().add_tensor(direction, mul=step))

        if cur_loss <= start_loss + c1 * step * dot_same_shape(direction, start_grad) and \
           -dot_same_shape(direction, cur_grad) <= -c2 * dot_same_shape(direction, start_grad):
           break

        step /= 2 
    
    return step


    
class LBFGSState(Enum):
    Finish = 0
    StillWorking = 1

class SparseLBFGS:
    lr: float # start length for the line search (default 1)
    history_size: int 
    param_groups: List[List[Parameter]]
    func: Callable[[], torch.Tensor] # must calculate loss
    func_grad: Callable[[], torch.Tensor] # must calculate loss and store gradients in parameters().grad
    tolerance_change: float
    tolerance_grad: float
    history: List[LBFGSHistory] # history for each param group
    state: LBFGSState

    __current_group_grads: List[ViewAsFlatTensor]
    __current_group_points: List[ViewAsFlatTensor]
    __current_loss: torch.Tensor

    def __init__(
            self, 
            param_groups: List[List[Parameter]],
            func: Callable[[], torch.Tensor],
            func_grad: Callable[[], torch.Tensor],
            lr = 1.0,
            history_size = 20,
            tolerance_change = 1e-9,
            tolerance_grad = 1e-7
        ) -> None:

        self.param_groups = param_groups
        self.func = func
        self.func_grad = func_grad
        self.lr = lr
        self.history_size = history_size
        self.tolerance_change = tolerance_change
        self.tolerance_grad = tolerance_grad
        
        self.history = [LBFGSHistory() for _ in range(len(param_groups))]

        self.state = LBFGSState.StillWorking

        self.__calc_func_grad()

    def __calc_func_grad(self):
        self.__current_loss = self.func_grad()
        self.__current_group_grads = self.__group_gradients()
        self.__current_group_points = self.__group_points()

    def __update_history(self,
                       point_delta: List[ViewAsFlatTensor], 
                       grad_delta: List[ViewAsFlatTensor]):
        assert len(point_delta) == len(self.param_groups)
        assert len(grad_delta) == len(self.param_groups)
        for i in range(len(point_delta)):
            self.history[i].update_history(point_delta[i], grad_delta[i])

    def __group_gradients(self) -> List[ViewAsFlatTensor]:
        grads = []
        for group in self.param_groups:
            grads.append(ViewAsFlatTensor([p.grad for p in group]))
            for p in group:
                p.grad = None
        return grads

    def __group_points(self) -> List[ViewAsFlatTensor]:
        points = []
        for group in self.param_groups:
            points.append(ViewAsFlatTensor([p.data for p in group]))
        return points

    def __reshape_in_groups(self, point: ViewAsFlatTensor) -> List[ViewAsFlatTensor]:
        i = 0
        res = []
        for group in self.param_groups:
            current_res = []
            for p in group:
                current_res.append(point.tensors[i].view_as(p.data))
                i += 1
            res.append(ViewAsFlatTensor(current_res))
        return res

    def __apply_step(self, step: ViewAsFlatTensor):
        i = 0
        for group in self.param_groups:
            for p in group:
                p.data -= step.tensors[i].view_as(p.data)
                i += 1

    def optimization_step(self) -> LBFGSState:
        if self.state == LBFGSState.Finish:
            return self.state

        grad = concat(self.__current_group_grads)

        if grad.norm().item() < self.tolerance_grad:
            self.state = LBFGSState.Finish
            return self.state

        group_directions = [self.history[i].compute_descent_direction(self.__current_group_grads[i]) 
                                for i in range(len(self.param_groups))]
        direction = concat(group_directions)
        point = concat(self.__current_group_points)

        def func(point: ViewAsFlatTensor) -> Tuple[torch.Tensor, ViewAsFlatTensor]:
            i = 0
            for group in self.param_groups:
                for p in group:
                    p.data = point.tensors[i].view_as(p.data)
                    i += 1

            loss = self.func_grad()
            gradient = []

            for group in self.param_groups:
                for p in group:
                    gradient.append(p.grad)
            
            return loss, ViewAsFlatTensor(gradient)


        alpha = wolfe_search(point, self.__current_loss, grad, direction, self.lr, func)
        direction.mul_scalar(alpha)

        if direction.norm.item() < self.tolerance_change:
            self.state = LBFGSState.Finish
            return self.state

        self.__apply_step(direction)
        self.__calc_func_grad()

        grad_delta = grad
        grad_delta.mul_scalar(-1)
        grad_delta.add_tensor(concat(self.__current_group_grads))
        self.__update_history(
            self.__reshape_in_groups(direction.mul_scalar(-1)),
            self.__reshape_in_groups(grad_delta)
        )

        return self.state