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

    model_lbfgs = NNModel(in_features=features, num_classes=classes, hidden_layers=hidden_layer).to(device)

    lbfgs_stats = sparse_lbfgs_train(model_lbfgs, trainloader, device, max_eval=eval)
    # combined_stats = combined_train(model_combined, trainloader, device, max_eval=eval)
    # adam_stats = adam_train(model_adam, trainloader, device, max_eval=eval)

    # plot_stats(lbfgs_stats, adam_stats, combined_stats, f"TestNNModel_f{features}_c{classes}_s{samples}")
    

if __name__ == "__main__":
    comparing(features=50, classes=200, hidden_layer=[200], samples=10000, eval=20)