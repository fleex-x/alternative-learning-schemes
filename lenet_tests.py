from train_methods import *
from models import LeNet
from common import plot_stats
import torchvision
from torchvision import transforms

def main():

    transform_train = transforms.Compose([
        transforms.RandomGrayscale(0.2),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(30),
        transforms.RandomAdjustSharpness(0.4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = 512

    trainset_class = torchvision.datasets.CIFAR10(root='data/', train=True, download=True,transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset_class, batch_size=batch_size, shuffle=True, num_workers=6)
    testset = torchvision.datasets.CIFAR10(root='data/', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("mps")

    model_lbfgs = LeNet().to(device)
    # sz = 0
    # for p in model_lbfgs.parameters():
    #     mul = 1
    #     for x in p.size():
    #         mul *= x
    #     sz += mul
    lbfgs_stats = lbfgs_train(model_lbfgs, trainloader, device, max_eval=50, history_size=10)

    model_adam = LeNet().to(device)
    adam_stats = adam_train(model_adam, trainloader, device, max_eval=50)

    plot_stats(lbfgs_stats, adam_stats, "lenet_cifar10#1")

if __name__ == "__main__":
    main()