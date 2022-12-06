from sklearn import datasets
import torch
from models import *
from torch.utils.data import DataLoader
from train_methods import *
from common import plot_stats

def comparing(features, classes, hidden_layer, samples, eval=20):
    print(f"Current model features {features}, classes {classes}, hidden_layer {hidden_layer}, samples {samples}")

    X, y = datasets.make_blobs(n_samples=samples, centers=classes, n_features=features)
    device = torch.device("mps")

    dataset = CustomDataset(X, y)
    trainloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    model_lbfgs = NNModel(in_features=features, num_classes=classes).to(device)
    lbfgs_stats = lbfgs_train(model_lbfgs, trainloader, device, max_eval=eval)

    model_adam = NNModel(in_features=features, num_classes=classes).to(device)
    adam_stats = adam_train(model_adam, trainloader, device, max_eval=eval)

    plot_stats(lbfgs_stats, adam_stats, f"NNModel_f{features}_c{classes}_s{samples}")
    

if __name__ == "__main__":
    # args = [
    #     (50, 200, 200, 10000),
    #     (20, 10, 40, 10000),
    #     (30, 200, 40, 10000),
    #     (30, 200, 500, 10000),
    # ]
    # for features, classes, hidden_layer, samples in args:
    #     comparing(features, classes, hidden_layer, samples)
    comparing(features=300, classes=900, hidden_layer=400, samples=50000, eval=50)