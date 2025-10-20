import numpy as np
import json

import torch
import torch.utils.data as Data

import torchvision.utils
from torchvision import models

import torchvision.transforms as transforms

import matplotlib.pyplot as plt
# matplotlib inline
# from zo_attack import *
from zo_attack import pgd_attack, zo_attack, r_attack, rzo_attack, rd_attack, imshow
import zo_attack as _zo
# import cifar10_models as cm

import os

# CUDA usage

use_cuda = torch.cuda.is_available()
use_mps  = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
device   = torch.device("cuda" if use_cuda else ("mps" if use_mps else "cpu"))
print("Using device:", device, "| CUDA avail:", torch.cuda.is_available())

# keep zo_attack on the same device
import zo_attack as _zo
_zo.device = device

data_root = os.path.join(os.path.dirname(__file__), "data")

# CIFAR-10 stats
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# data transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# datasets (download once)
trainset = torchvision.datasets.CIFAR10(
    root=data_root, train=True, transform=transform_train, download=True
)
testset = torchvision.datasets.CIFAR10(
    root=data_root, train=False, transform=transform_test, download=True
)

# (optional) loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)



def main():
    # Data
    # # https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
    # class_idx = json.load(open("./data/imagenet_class_index.json"))
    # idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    #
    # transform = transforms.Compose([
    #     transforms.Resize((299, 299)),
    #     transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    # ])
    #
    # normal_data = image_folder_custom_label(root = './data/imagenet', transform = transform, custom_label = idx2label)
    # normal_loader = Data.DataLoader(normal_data, batch_size=1, shuffle=False)
    #
    # normal_iter = iter(normal_loader)
    # images, labels = normal_iter.next()
    #
    # print("True Image & True Label")
    # imshow(torchvision.utils.make_grid(images, normalize=True), [normal_data.classes[i] for i in labels])
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    transform_validation = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                    [0.2023, 0.1994, 0.2010])])

    transform_validation = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, transform=transform_train, download=True)
    testset  = torchvision.datasets.CIFAR10(root=data_root, train=False, transform=transform_validation, download=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Load model
    # model = cm.vgg11_bn(pretrained=True).to(device)
    model = torch.hub.load('chenyaofo/pytorch-cifar-models',
                       'cifar10_vgg11_bn', pretrained=True).to(device)

    # get some random training images
    dataiter = iter(testloader)
 
    # Attack
    print("Attack Image & Predicted Label")

    model.eval()

    correct = 0
    total = 0

    for images, labels in dataiter:
        total += 1
        # print("Processing image number: %s" % total)
        # if total == 9933:
        if total == 1:
            # show image and label
            images = images.to(device)
            labels = labels.to(device)
            print("True Image & Predicted Label")
            outputs = model(images)
            _, pre = torch.max(outputs.data, 1)
            imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), [classes[i] for i in pre],'Original')
            print("True label: ", [classes[i] for i in labels.cpu().data.numpy()])

            # PGD attack
            ori_images = images
            images, i1, val_perturb_pgd, time_perturb_1 = pgd_attack(model, images, labels)
            labels = labels.to(device)
            outputs = model(images)

            _, pre = torch.max(outputs.data, 1)

            total += 1
            correct += (pre == labels).sum()

            imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), [classes[i] for i in pre],'PGD')
            distance = images.cpu() - ori_images.cpu()
            print('PGD-attack: Distance of the attack example to original image:', np.linalg.norm(distance.cpu().data.numpy()))
            ori_images = ori_images.to(device)
            eta = images - ori_images

            # # ZO attack
            # images, i2, val_perturb_zo, time_perturb_2 = zo_attack(model, ori_images, labels)
            # labels = labels.to(device)
            # outputs = model(images)

            # _, pre = torch.max(outputs.data, 1)

            # total += 1
            # correct += (pre == labels).sum()

            # imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), [classes[i] for i in pre],'ZO')
            # distance = images.cpu() - ori_images.cpu()
            # print('ZO-attack: Distance of the attack example to original image:',
            #       np.linalg.norm(distance.cpu().data.numpy()))

            # Riemannian attack
            images, i3, val_perturb_r, time_perturb_3 = r_attack(model, ori_images, eta, labels)
            labels = labels.to(device)
            outputs = model(images)

            _, pre = torch.max(outputs.data, 1)

            total += 1
            correct += (pre == labels).sum()

            imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), [classes[i] for i in pre],'RGD')
            distance = images.cpu() - ori_images.cpu()
            print('R-attack: Distance of the attack example to original image:', np.linalg.norm(distance.cpu().data.numpy()))

            # Riemannian ZO attack
            images, i4, val_perturb_rzo, time_perturb_4 = rzo_attack(model, ori_images, labels, m=500)
            labels = labels.to(device)
            outputs = model(images)

            _, pre = torch.max(outputs.data, 1)

            total += 1
            correct += (pre == labels).sum()

            imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), [classes[i] for i in pre],'RZO')
            distance = images.cpu() - ori_images.cpu()
            print('RZO-attack: Distance of the attack example to original image:',
                  np.linalg.norm(distance.cpu().data.numpy()))

            # Riemannian Dueling attack
            images, i5, val_perturb_rd, time_perturb_5 = rd_attack(model, ori_images, labels, m=500)
            labels = labels.to(device)
            outputs = model(images)

            _, pre = torch.max(outputs.data, 1)

            total += 1
            correct += (pre == labels).sum()

            imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), [classes[i] for i in pre],'RDueling')
            distance = images.cpu() - ori_images.cpu()
            print('RDueling-attack: Distance of the attack example to original image:',
                  np.linalg.norm(distance.cpu().data.numpy()))

            break

    save_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots()
    plt.xscale('symlog')
    ax.plot(range(i1), val_perturb_pgd[:i1], 'r--', label='White-box PGD')
    # ax.plot(range(i2), val_perturb_zo[:i2], 'g--', label='Black-box ZO PGD')
    ax.plot(range(i3), val_perturb_r[:i3], 'k--',label='White-box Riemannian')
    ax.plot(range(i4), val_perturb_rzo[:i4], 'b:',label='Black-box ZO Riemannian')
    ax.plot(range(i5), val_perturb_rd[:i5], 'g:',label='Black-box Dueling Riemannian')
    ax.set(xlabel='Number of iteration', ylabel='Loss value',
           title='Loss value')
    ax.legend(loc='upper left')
    ax.grid()
    # Save figure to pwd/results
    plot_path1 = os.path.join(save_dir, "loss_vs_iteration.png")
    plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path1}")
    plt.close(fig)   # close to free memory

    fig, ax = plt.subplots()
    plt.xscale('symlog')
    ax.plot(time_perturb_1[:i1], val_perturb_pgd[:i1], 'r--', label='White-box PGD')
    # ax.plot(time_perturb_2[:i2], val_perturb_zo[:i2], 'g--', label='Black-box ZO PGD')
    ax.plot(time_perturb_3[:i3], val_perturb_r[:i3], 'k--', label='White-box Riemannian')
    ax.plot(time_perturb_4[:i4], val_perturb_rzo[:i4], 'b:', label='Black-box ZO Riemannian')
    ax.plot(time_perturb_5[:i5], val_perturb_rd[:i5], 'g:',label='Black-box Dueling Riemannian')
    ax.set(xlabel='CPU time', ylabel='Loss value',
           title='Loss value')
    ax.legend(loc='upper left')
    ax.grid()
    plot_path2 = os.path.join(save_dir, "loss_vs_time.png")
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path2}")
    plt.close(fig)



    # print('Accuracy of test text: %f %%' % (100 * float(correct) / total))


if __name__== "__main__":
    main()
