import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import json
import torch
import time
import os

## CUDA usage
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

# imagenet data functions
class_idx = json.load(open("./data/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]


def image_folder_custom_label(root, transform, custom_label):
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']

    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes

    label2idx = {}

    for i, item in enumerate(idx2label):
        label2idx[item] = i

    new_data = dsets.ImageFolder(root=root, transform=transform,
                                 target_transform=lambda x: custom_label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data


# cifar10 dataloader
def make_dataloaders(params):
    """
    Make a Pytorch dataloader object that can be used for traing and valiation
    Input:
        - params dict with key 'path' (string): path of the dataset folder
        - params dict with key 'batch_size' (int): mini-batch size
        - params dict with key 'num_workers' (int): number of workers for dataloader
    Output:
        - trainloader and testloader (pytorch dataloader object)
    """
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    transform_validation = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                                    [0.2023, 0.1994, 0.2010])])

    transform_validation = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root=params['path'], train=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=params['path'], train=False, transform=transform_validation)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=params['batch_size'], shuffle=False, num_workers=4)
    return trainloader, testloader


def imshow(img, label, idx, methodName):
    save_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(save_dir, exist_ok=True)

    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(3, 3.3))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.suptitle(label, fontsize=20)

    # plt.show()
    # save figure
    save_path = os.path.join(save_dir, f"{idx}_{methodName}_{label}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved image to {save_path}")

    plt.close()


# projection gradient attack
def pgd_attack(model, images, labels, eps=0.05, alpha=1e-4, iters=150):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
    ori_images = images.data
    max_norm = eps * torch.norm(ori_images)

    val_perturb = np.zeros(iters)
    time_perturb = np.zeros(iters)

    print("PGD attack running, maximum perturbation norm", max_norm.cpu().data.numpy())
    start = time.time()
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        _, pre = torch.max(outputs.data, 1)
        if torch.eq(pre, labels) == 0:
            return images, i, val_perturb, time_perturb

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        pertubed_direction = alpha * images.grad.sign()
        adv_images = images + pertubed_direction

        eta = adv_images - ori_images
        if torch.norm(eta) > max_norm:
            images = ori_images + eta / torch.norm(eta) * max_norm
        else:
            images = adv_images
        images = images.detach_()
        print(
            "PGD attack iter: %s, loss function value: %f, distance from original image: %f, current label: %s" % (
                i, cost.cpu().data.numpy(), torch.norm(images - ori_images).cpu().data.numpy(),
                pre[0].cpu().data.numpy()))

        val_perturb[i] = cost.cpu().data.numpy()
        time_perturb[i] = time.time() - start

    return images, i, val_perturb, time_perturb


# riemannian attack
def r_attack(model, images, eta, labels, alpha=2e-3, iters=150):
    # eta is the original pertubation
    # the starting point is actually images+eta
    max_norm = torch.norm(eta)
    ori_images = images.data
    # images = images + eta
    eta = torch.randn(images.size()).to(device)
    eta = eta / torch.norm(eta) * max_norm
    images = images + eta
    images = images.detach_()
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    val_perturb = np.zeros(iters)
    time_perturb = np.zeros(iters)

    print("Riemannian attack running, maximum pertubation norm", max_norm.cpu().data.numpy())
    start = time.time()
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        _, pre = torch.max(outputs.data, 1)
        if torch.eq(pre, labels) == 0:
            return images, i, val_perturb, time_perturb

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        # update
        rgrad = images.grad - torch.sum(torch.mul(images.grad, eta)) * eta / max_norm ** 2
        adv_images = images + alpha * rgrad

        eta = adv_images - ori_images
        eta = eta / torch.norm(eta) * max_norm
        images = ori_images + eta
        images = images.detach_()
        print("Riemannian attack iter: %s, loss function value: %f, distance from original image: %f" % (
            i, cost.cpu().data.numpy(), torch.norm(images - ori_images).cpu().data.numpy()))

        val_perturb[i] = cost.cpu().data.numpy()
        time_perturb[i] = time.time() - start

    return images, i, val_perturb, time_perturb


# zeros order attack
def zo_attack(model, images, labels, eps=0.05, iters=1500, alpha=5e-4, m=500, mu=1e-6):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
    ori_images = images.data
    max_norm = eps * torch.norm(ori_images)

    val_perturb = np.zeros(iters)
    time_perturb = np.zeros(iters)

    print("ZO attack running, maximum perturbation norm", max_norm.cpu().data.numpy())
    start = time.time()
    for k in range(iters):
        outputs = model(images)
        _, pre = torch.max(outputs.data, 1)
        if torch.eq(pre, labels) == 0:
            return images, k, val_perturb, time_perturb

        # model.zero_grad()
        cost = loss(outputs, labels).to(device)
        # cost.backward()

        # zeroth-order estimation of Euclidean gradient
        grad_estimate = torch.zeros(images.size()).to(device)
        for j in range(m):
            temp = torch.randn(images.size()).to(device)
            temp = temp.to(device)
            temp_image = ori_images + mu * temp
            outputs = model(temp_image)
            cost_temp = loss(outputs, labels).to(device)
            grad_estimate += (cost_temp - cost) / mu * temp
            grad_estimate = grad_estimate.detach_()
        grad_estimate /= m
        grad_estimate = grad_estimate.detach_()

        pertubed_direction = alpha * grad_estimate
        # adv_images = images + pertubed_direction

        val_perturb[k] = cost.cpu().data.numpy()
        time_perturb[k] = time.time() - start
        if torch.norm(pertubed_direction) > max_norm:
            images = ori_images + pertubed_direction / torch.norm(pertubed_direction) * max_norm
        else:
            images = ori_images + pertubed_direction

        print("ZO attack iteration: %s, loss function value: %f, distance from original image: %f" % (
            k, val_perturb[k], torch.norm(images - ori_images).cpu().data.numpy()))

    return images, k, val_perturb, time_perturb


# zeros order Riemannian attack
def rzo_attack(model, images, labels, eps=0.05, alpha=1e-3, pre_alpha=0.1, pre_iter=1000, pre_m=100, iters=1000, mu=1e-6,
               m=500):
    # eta is the original pertubation
    # the starting point is actually images+eta
    ori_images = images.data
    # eta = torch.randn(images.size()).to(device)
    # eta = eta / torch.norm(eta) * max_norm
    # images = images + eta
    images = images.detach_()
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
    max_norm = eps * torch.norm(ori_images)

    val_perturb = np.zeros(pre_iter + iters)
    time_perturb = np.zeros(pre_iter + iters)
    k = 0

    print("Riemannian ZO attack running, maximum perturbation norm", max_norm.cpu().data.numpy())
    print("Scheme 1: escaping from the original image")
    start = time.time()
    for k in range(pre_iter):
        eta = torch.zeros(images.size()).to(device)
        outputs = model(images)
        _, pre = torch.max(outputs.data, 1)
        # if torch.eq(pre, labels) == 0:
        #     return images, k, val_perturb

        # model.zero_grad()
        cost = loss(outputs, labels).to(device)
        # cost.backward()

        # zeroth-order estimation of Riemannian gradient
        grad_estimate = torch.zeros(images.size()).to(device)
        for j in range(pre_m):
            temp = torch.randn(images.size()).to(device)
            temp_eta = eta + mu * temp
            temp_eta = temp_eta.to(device)
            temp_image = ori_images + temp_eta
            outputs = model(temp_image)
            cost_temp = loss(outputs, labels).to(device)
            grad_estimate = grad_estimate + (cost_temp - cost) / mu * temp
            grad_estimate = grad_estimate.detach_()
        grad_estimate = grad_estimate / pre_m
        grad_estimate = grad_estimate.detach_()

        pertubed_direction = pre_alpha * grad_estimate / torch.norm(grad_estimate)
        adv_images = images + pertubed_direction

        eta = adv_images - ori_images

        if torch.norm(eta) > max_norm:
            images = ori_images + eta / torch.norm(eta) * max_norm
            break
        else:
            images = ori_images + eta
        val_perturb[k] = cost.cpu().data.numpy()
        time_perturb[k] = time.time() - start

        print("RZO pre_attack iteration: %s, loss function value: %f, distance from original image: %f" % (
            k, val_perturb[k], torch.norm(images - ori_images).cpu().data.numpy()))

    k = k + 1

    print("Scheme 2: optimization on the ball")
    for i in range(iters):
        # images.requires_grad = True
        outputs = model(images)
        _, pre = torch.max(outputs.data, 1)
        if torch.eq(pre, labels) == 0:
            return images, k + i, val_perturb, time_perturb

        # model.zero_grad()
        cost = loss(outputs, labels).to(device)
        # cost.backward()

        # zeroth-order estimation of Riemannian gradient
        grad_estimate = torch.zeros(images.size()).to(device)
        for j in range(m):
            temp = torch.randn(images.size()).to(device)
            temp = temp - torch.sum(torch.mul(temp, eta)) * eta / torch.norm(eta) ** 2  # projection
            temp_eta = (eta + mu * temp) / torch.norm(eta + mu * temp) * max_norm  # retraction
            temp_eta = temp_eta.to(device)
            temp_image = ori_images + temp_eta
            outputs = model(temp_image)
            cost_temp = loss(outputs, labels).to(device)
            grad_estimate = grad_estimate + (cost_temp - cost) / mu * temp
            grad_estimate = grad_estimate.detach_()
        grad_estimate = grad_estimate / m
        grad_estimate = grad_estimate.detach_()

        # update
        adv_images = images + alpha * grad_estimate

        eta = adv_images - ori_images
        eta = eta / torch.norm(eta) * max_norm
        images = ori_images + eta
        images = images.detach_()
        print("RZO attack iter: %s, loss function value: %f, distance from original image: %f" % (
            i, cost.cpu().data.numpy(), torch.norm(images - ori_images).cpu().data.numpy()))

        val_perturb[k + i] = cost.cpu().data.numpy()
        time_perturb[k + i] = time.time() - start



    return images, k + i, val_perturb, time_perturb

#Riemannian Dueling attack
def rd_attack(model, images, labels, eps=0.05, alpha=1e-1, pre_alpha=0.5, pre_iter=1000, pre_m=10, iters=1000, mu=1e-6,
               m=500):
    # eta is the original pertubation
    # the starting point is actually images+eta
    ori_images = images.data
    # eta = torch.randn(images.size()).to(device)
    # eta = eta / torch.norm(eta) * max_norm
    # images = images + eta
    images = images.detach_()
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
    max_norm = eps * torch.norm(ori_images)

    val_perturb = np.zeros(pre_iter + iters)
    time_perturb = np.zeros(pre_iter + iters)
    k = 0

    best = float("-inf")
    start = time.time()


    print("Riemannian Dueling attack running, maximum perturbation norm", max_norm.cpu().data.numpy())
    print("Scheme 1: escaping from the original image")
    for k in range(pre_iter):
        eta = torch.zeros(images.size()).to(device)
        outputs = model(images)
        _, pre = torch.max(outputs.data, 1)
        # if torch.eq(pre, labels) == 0:
        #     return images, k, val_perturb

        # model.zero_grad()
        cost = loss(outputs, labels).to(device)
        # cost.backward()

        # zeroth-order estimation of Riemannian gradient
        grad_estimate = torch.zeros(images.size()).to(device)
        for j in range(pre_m):
            # temp = torch.randn(images.size()).to(device)
            # temp_eta = eta + mu * temp
            # temp_eta = temp_eta.to(device)
            # temp_image = ori_images + temp_eta
            # outputs = model(temp_image)
            # cost_temp = loss(outputs, labels).to(device)
            # grad_estimate = grad_estimate + (cost_temp - cost) / mu * temp
            # grad_estimate = grad_estimate.detach_()
            
            temp = torch.randn(images.size()).to(device)
            dims = tuple(range(1, temp.ndim))
            nrm  = torch.linalg.vector_norm(temp, ord=2, dim=dims, keepdim=True)
            temp = temp / (nrm.clamp_min(1e-12))

            temp_eta1 = eta + mu * temp
            temp_eta1 = temp_eta1.to(device)
            temp_image1 = ori_images + temp_eta1
            outputs1 = model(temp_image1)
            cost_temp1 = loss(outputs1, labels).to(device)

            temp_eta2 = eta - mu * temp
            temp_eta2 = temp_eta2.to(device)
            temp_image2 = ori_images + temp_eta2
            outputs2 = model(temp_image2)
            cost_temp2 = loss(outputs2, labels).to(device)  

            sign = torch.where(
                cost_temp1 > cost_temp2,
                torch.tensor(1.0, device=temp.device, dtype=temp.dtype),
                torch.tensor(-1.0, device=temp.device, dtype=temp.dtype),
            )
            grad_estimate = grad_estimate + sign * temp
            grad_estimate = grad_estimate.detach_()
        grad_estimate = grad_estimate / pre_m
        grad_estimate = grad_estimate.detach_()

        pertubed_direction = pre_alpha * grad_estimate / torch.norm(grad_estimate)
        adv_images = images + pertubed_direction

        eta = adv_images - ori_images

        if torch.norm(eta) > max_norm:
            images = ori_images + eta / torch.norm(eta) * max_norm
            break
        else:
            images = ori_images + eta

        # keep best-so-far
        best = max(best, cost.detach().item())
        val_perturb[k] = best
        time_perturb[k] = time.time() - start

        print("RDueling pre_attack iteration: %s, loss function value: %f, distance from original image: %f" % (
            k, val_perturb[k], torch.norm(images - ori_images).cpu().data.numpy()))

    # k = k + 1

    print("Scheme 2: optimization on the ball")
    for i in range(iters):
        # images.requires_grad = True
        outputs = model(images)
        _, pre = torch.max(outputs.data, 1)
        if torch.eq(pre, labels) == 0:
            return images, k + i, val_perturb, time_perturb

        # model.zero_grad()
        cost = loss(outputs, labels).to(device)
        # cost.backward()

        # zeroth-order estimation of Riemannian gradient
        grad_estimate = torch.zeros(images.size()).to(device)
        for j in range(m):
            temp = torch.randn(images.size()).to(device)
            temp = temp - torch.sum(torch.mul(temp, eta)) * eta / torch.norm(eta) ** 2  # projection
            dims = tuple(range(1, temp.ndim))
            nrm  = torch.linalg.vector_norm(temp, ord=2, dim=dims, keepdim=True)
            temp = temp / (nrm.clamp_min(1e-12))

            # trace_AtA = (temp ** 2).sum()
            # print(f"trace(A^T A) = {trace_AtA.item():.6f}")

            temp_eta1 = (eta + mu * temp) / torch.norm(eta + mu * temp) * max_norm  # retraction
            temp_eta1 = temp_eta1.to(device)
            temp_image1 = ori_images + temp_eta1
            outputs1 = model(temp_image1)
            cost_temp1 = loss(outputs1, labels).to(device)

            temp_eta2 = (eta - mu * temp) / torch.norm(eta - mu * temp) * max_norm  # retraction
            temp_eta2 = temp_eta2.to(device)
            temp_image2 = ori_images + temp_eta2
            outputs2 = model(temp_image2)
            cost_temp2 = loss(outputs2, labels).to(device)  

            # grad_estimate = grad_estimate + (cost_temp - cost) / mu * temp
            sign = torch.where(
                cost_temp1 > cost_temp2,
                torch.tensor(1.0, device=temp.device, dtype=temp.dtype),
                torch.tensor(-1.0, device=temp.device, dtype=temp.dtype),
            )
            grad_estimate = grad_estimate + sign * temp
            grad_estimate = grad_estimate.detach_()

        grad_estimate = grad_estimate / m
        dims = tuple(range(1, grad_estimate.ndim))
        nrm  = torch.linalg.vector_norm(grad_estimate, ord=2, dim=dims, keepdim=True)
        grad_estimate = grad_estimate / (nrm.clamp_min(1e-12))
        grad_estimate = grad_estimate.detach_()

        # update
        adv_images = images + alpha * grad_estimate

        eta = adv_images - ori_images
        eta = eta / torch.norm(eta) * max_norm
        images = ori_images + eta
        images = images.detach_()
        print("RDueling attack iter: %s, loss function value: %f, distance from original image: %f" % (
            i, cost.cpu().data.numpy(), torch.norm(images - ori_images).cpu().data.numpy()))

        # keep best-so-far
        best = max(best, cost.detach().item())
        val_perturb[k + i] = best
        time_perturb[k + i] = time.time() - start
        
    return images, k + i, val_perturb, time_perturb


def rd_attack_new(model, images, eta, labels, alpha=0.05, iters=2000, m=100, mu=1e-6):
    # eta is the original pertubation
    # the starting point is actually images+eta
    max_norm = torch.norm(eta)
    ori_images = images.data
    # images = images + eta
    eta = torch.randn(images.size()).to(device)
    eta = eta / torch.norm(eta) * max_norm
    images = images + eta
    images = images.detach_()
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    val_perturb = np.zeros(iters)
    time_perturb = np.zeros(iters)

    best = float("-inf")
    start = time.time()

    print("Riemannian Dueling attack running, maximum perturbation norm", max_norm.cpu().data.numpy())
    for k in range(iters):
        # images.requires_grad = True
        outputs = model(images)
        _, pre = torch.max(outputs.data, 1)
        if torch.eq(pre, labels) == 0:
            return images, k, val_perturb, time_perturb

        # model.zero_grad()
        cost = loss(outputs, labels).to(device)
        # cost.backward()

        # zeroth-order estimation of Riemannian gradient
        grad_estimate = torch.zeros(images.size()).to(device)
        for j in range(m):
            temp = torch.randn(images.size()).to(device)
            temp = temp - torch.sum(torch.mul(temp, eta)) * eta / torch.norm(eta) ** 2  # projection
            dims = tuple(range(1, temp.ndim))
            nrm  = torch.linalg.vector_norm(temp, ord=2, dim=dims, keepdim=True)
            temp = temp / (nrm.clamp_min(1e-12))

            # trace_AtA = (temp ** 2).sum()
            # print(f"trace(A^T A) = {trace_AtA.item():.6f}")

            temp_eta1 = (eta + mu * temp) / torch.norm(eta + mu * temp) * max_norm  # retraction
            temp_eta1 = temp_eta1.to(device)
            temp_image1 = ori_images + temp_eta1
            outputs1 = model(temp_image1)
            cost_temp1 = loss(outputs1, labels).to(device)

            temp_eta2 = (eta - mu * temp) / torch.norm(eta - mu * temp) * max_norm  # retraction
            temp_eta2 = temp_eta2.to(device)
            temp_image2 = ori_images + temp_eta2
            outputs2 = model(temp_image2)
            cost_temp2 = loss(outputs2, labels).to(device)  

            sign = torch.where(
                cost_temp1 > cost_temp2,
                torch.tensor(1.0, device=temp.device, dtype=temp.dtype),
                torch.tensor(-1.0, device=temp.device, dtype=temp.dtype),
            )
            grad_estimate = grad_estimate + sign * temp
            grad_estimate = grad_estimate.detach_()

        grad_estimate = grad_estimate / m
        dims = tuple(range(1, grad_estimate.ndim))
        nrm  = torch.linalg.vector_norm(grad_estimate, ord=2, dim=dims, keepdim=True)
        grad_estimate = grad_estimate / (nrm.clamp_min(1e-12))
        grad_estimate = grad_estimate.detach_()

        # update
        adv_images = images + alpha * grad_estimate

        eta = adv_images - ori_images
        eta = eta / torch.norm(eta) * max_norm
        images = ori_images + eta
        images = images.detach_()
        print("RDueling attack iter: %s, loss function value: %f, distance from original image: %f" % (
            k, cost.cpu().data.numpy(), torch.norm(images - ori_images).cpu().data.numpy()))

        # keep best-so-far
        best = max(best, cost.detach().item())
        val_perturb[k] = best
        time_perturb[k] = time.time() - start
        
    return images, k, val_perturb, time_perturb

