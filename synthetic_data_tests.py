from sklearn import datasets
import torch
from models import *
from torch.utils.data import DataLoader
from train_methods import *
import matplotlib.pyplot as plt

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

def comparing(features, classes, hidden_layer, samples):
    print(f"Current model features {features}, classes {classes}, hidden_layer {hidden_layer}, samples {samples}")

    X, y = datasets.make_blobs(n_samples=samples, centers=classes, n_features=features)
    device = torch.device("mps")

    dataset = CustomDataset(X, y)
    trainloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    model_lbfgs = TwoLayersModel(in_features=features, num_classes=classes, hidden_layer=hidden_layer).to(device)
    lbfgs_stats = lbfgs_train(model_lbfgs, trainloader, device, max_eval=20)

    model_adam = TwoLayersModel(in_features=features, num_classes=classes).to(device)
    adam_stats = adam_train(model_adam, trainloader, device, max_eval=20)

    plot_stats(lbfgs_stats, adam_stats, f"f{features}_c{classes}_h{hidden_layer}_s{samples}")



if __name__ == "__main__":
    args = [
        (50, 200, 200, 10000),
        (20, 10, 40, 10000),
        (20, 10, 40, 10000),
        (30, 120, 40, 10000),
        (30, 120, 500, 10000)
    ]
    for features, classes, hidden_layer, samples in args:
        comparing(features, classes, hidden_layer, samples)