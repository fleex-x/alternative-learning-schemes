import matplotlib.pyplot as plt
from train_methods import TrainingStats
from torch import nn 

def model_size(model: nn.Module):
    sz = 0
    for p in filter(lambda p: p.requires_grad , model.parameters()):
        mul = 1
        for x in p.size():
            mul *= x
        sz += mul
    return sz

def plot_stats(lbfgs_stats: TrainingStats, 
               adam_stats: TrainingStats,
               name: str):
    plt.plot([i for i in range(len(lbfgs_stats.loss))], lbfgs_stats.loss, label="lbfgs")
    plt.plot([i for i in range(len(adam_stats.loss))], adam_stats.loss, label="adam")
    plt.ylabel("loss")
    plt.xlabel("function evaluations")
    plt.yscale("log")
    plt.title("loss plot")
    plt.legend()
    plt.savefig(fname=f"plots/loss/{name}.png")
    plt.cla()

    plt.plot([i for i in range(len(lbfgs_stats.grad_norm))], lbfgs_stats.grad_norm, label="lbfgs")
    plt.plot([i for i in range(len(adam_stats.grad_norm))], adam_stats.grad_norm, label="adam")
    plt.ylabel("grad norm")
    plt.xlabel("function evaluations")
    plt.yscale("log")
    plt.title("grad norm plot")
    plt.legend()
    plt.savefig(fname=f"plots/grad/{name}.png")
    plt.cla()

    plt.plot([i for i in range(len(lbfgs_stats.train_accuracy))], lbfgs_stats.train_accuracy, label="lbfgs")
    plt.plot([i for i in range(len(adam_stats.train_accuracy))], adam_stats.train_accuracy, label="adam")
    plt.ylabel("train accuracy")
    plt.xlabel("function evaluations")
    plt.title("train accuracy plot")
    plt.legend()
    plt.savefig(fname=f"plots/accuracy/{name}.png")
    plt.cla()