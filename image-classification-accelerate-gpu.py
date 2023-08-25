import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from accelerate import Accelerator
from accelerate.utils import set_seed
import bitsandbytes as bnb
# from efficientnet_pytorch import EfficientNet
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# from tqdm.notebook import trange
import os
from multiprocessing import cpu_count
from argparse import ArgumentParser
from typing import Union, OrderedDict
import random

try:
    from lion_pytorch import Lion
except:
    print("Install lion_pytorch if you want to use Lion optimizer")

print(f"PyTorch version: {torch.__version__}")
print(f"CPU cores: {cpu_count()}")


def get_device(accelerator):
    return accelerator.device

def get_model(opts):
    model = None
    if opts.network == "resnet18":
        model = models.resnet18(True)
        # Create a new last layer.
        model.fc = nn.Linear(
            in_features=model.fc.in_features, 
            out_features=opts.num_classes,
            bias=model.fc.bias is not None
        )
    elif opts.network == "resnet34":
        model = models.resnet34(True)
        # Create a new last layer.
        model.fc = nn.Linear(
            in_features=model.fc.in_features, 
            out_features=opts.num_classes,
            bias=model.fc.bias is not None
        )
    elif opts.network == "resnet50":
        model = models.resnet50(True)
        # Create a new last layer.
        model.fc = nn.Linear(
            in_features=model.fc.in_features, 
            out_features=opts.num_classes, 
            bias=model.fc.bias is not None
        )
    elif opts.network == "resnet101":
        model = models.resnet101(True)
        # Create a new last layer.
        model.fc = nn.Linear(
            in_features=model.fc.in_features, 
            out_features=opts.num_classes, 
            bias=model.fc.bias is not None
        )
    elif opts.network == "resnet152":
        model = models.resnet152(True)
        # Create a new last layer.
        model.fc = nn.Linear(
            in_features=model.fc.in_features, 
            out_features=opts.num_classes, 
            bias=model.fc.bias is not None
        )
    elif opts.network == "alexnet":
        model = models.alexnet(True)
        # Create a new last layer.
        model.classifier[6] = nn.Linear(
            in_features=model.classifier[6].in_features, 
            out_features=opts.num_classes,
            bias=model.classifier[6].bias
        )
    elif opts.network == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(True)
    elif opts.network == "efficientnet_b0":
        model = models.efficientnet_b0(True)
    elif opts.network == "efficientnet_b5":
        model = models.efficientnet_b5(True)
    elif opts.network == "efficientnet_b7":
        model = models.efficientnet_b7(True)
        # Create a new last layer.
        # model.

    return model

def get_optimizer(opts, model):
    if opts.optim_name == "adam":
        return optim.Adam(model.parameters(), lr=opts.learning_rate)
    elif opts.optim_name == "adamw":
        return optim.AdamW(model.parameters(), lr=opts.learning_rate)
        # return bnb.optim.AdamW8bit(model.parameters(), lr=opts.learning_rate)
    elif opts.optim_name == "lion":
        return Lion(model.parameters(), lr=opts.learning_rate / opts.learning_rate_downscale, use_triton=opts.use_triton)

def get_dataloader(opts):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((opts.image_size, opts.image_size)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if opts.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root='.', 
            train=True,  
            download=True, 
            transform=transform
        )
    elif opts.dataset == "oxfordiiitpet":
        train_dataset = datasets.OxfordIIITPet(
            root='.',
            split='trainval',
            download=True,
            transform=transform
        )
    elif opts.dataset == "fakedata":
        train_dataset = datasets.FakeData(
            size=opts.num_images,
            image_size=(3, opts.image_size, opts.image_size),
            num_classes=opts.num_classes,
            transform=transform
        )
    trainloader = DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=int(opts.workers),
        drop_last=True,
        persistent_workers=opts.persistent_workers
    )
    return trainloader

def get_loss_fn():
    pass

def train(opts, accelerator, losses):

    # get everything!
    device     = get_device(accelerator)
    dataloader = get_dataloader(opts)
    model      = get_model(opts)
    optimizer  = get_optimizer(opts, model)

    print(f"device: {device}")
    print(f"dataloader: {len(dataloader)}")
    
    # Prepare for Accelerate
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Move model to device
    model.to(device)

    # Start training
    for epoch in range(opts.max_epochs):
        # loss = 0.0
        # for batch_idx, inputs, targets in tqdm(enumerate(dataloader), desc=f"[Epoch: {epoch} | Loss: {loss}]"):
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"[Epoch: {epoch}]")):

            # Forward pass and compute loss.
            inputs, targets = batch
            preds = model(inputs)
            loss = F.cross_entropy(preds, targets)
            losses.append(loss.item())

            # Downscale the loss to account for accumulation steps.
            loss /= opts.gradient_accumulation_steps
            accelerator.backward(loss)

            # Reference: https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html#3.-How-to-make-it-work
            # It's time to update the weights using accumulated gradients.
            if ((batch_idx + 1) % opts.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(dataloader)):
                # Now that the gradients were being accumulated by optimizer,
                # just update the weights now.
                optimizer.step()

                # Also set the opimizer gradients to zero again. :)
                optimizer.zero_grad()
    return losses

def get_opts():

    # We set random seed, if not mentioned.
    random_seed = random.randint(1, 1_000_000)

    parser = ArgumentParser()
    parser.add_argument(
        "--seed", 
        default=random_seed, 
        type=int, 
        help="Set the seed for reproducible experiments."
    )
    parser.add_argument(
        "--max_epochs", 
        default=50, 
        type=int, 
        help="Set maximum epochs for training."
    )
    parser.add_argument(
        '--accelerator', 
        default='auto', 
        type=str, 
        help='Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”) as well as custom accelerator instances'
    )
    parser.add_argument(
        '--devices', 
        default=None, 
        type=int, 
        help='Will be mapped to either gpus, tpu_cores, num_processes or ipus, based on the accelerator type'
    )
    parser.add_argument(
        '--strategy', 
        default=None, 
        type=str, 
        help='Strategy controls the model distribution across training, evaluation, and prediction to be used by the Trainer'
    )
    parser.add_argument(
        '--precision', 
        default=32, 
        type=Union[int, str], 
        help='Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16). Can be used on CPU, GPU, TPUs, HPUs or IPUs. Default: 32'
    )
    parser.add_argument(
        '--gradient_accumulation_steps', 
        default=1, 
        type=int, 
        help='Accumulates grads every k batches or as set up in the dict. Default: 1'
    )
    parser.add_argument(
        "--network", 
        default="resnet50", 
        type=str
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        help="Select of the following datasets: cifar10, fakedata."
    )
    parser.add_argument(
        "--num_images", 
        default=1_000_000, 
        type=int
    )
    parser.add_argument(
        "--num_classes", 
        default=1000, 
        type=int
    )
    parser.add_argument(
        '--batch_size', 
        default=16, 
        type=int, 
        help='Batch size for training'
    )
    parser.add_argument(
        "--image_size", 
        default=224, 
        type=int, 
        help="The image size of dataset."
    )
    parser.add_argument(
        '--workers', 
        default=cpu_count(), 
        type=int, 
        help='Number of train dataloader workers'
    )
    parser.add_argument(
        '--persistent_workers', 
        default=False, 
        action="store_true"
    )
    parser.add_argument(
        '--learning_rate', 
        default=1e-3, 
        type=float, 
        help='Optimizer learning rate'
    )
    parser.add_argument(
        '--learning_rate_downscale', 
        default=3, 
        type=float, 
        help='The value by which learning rate should be scaled down. (Only applicable for LION optimizer)'
    )
    parser.add_argument(
        '--optim_name', 
        default='adam', 
        type=str, 
        help='Which optimizer to use: adam, adamw, lion'
    )
    parser.add_argument(
        '--use_triton', 
        default=False,
        action="store_true", 
        help='Use triton for Lion to use CUDA kernels'
    )
    parser.add_argument(
        "--reload_dataloaders_every_n_epochs", 
        default=1, 
        type=int, 
        help="Reloads dataloaders after what epoch?"
    )
    opts = parser.parse_args()
    return opts

def plot_and_save(losses, opts):

    # naming.
    name = f'./losses-optim={opts.optim_name}-lr={opts.learning_rate}'

    # Save loss array.
    np.save(f'{name}.npy', np.array(losses))

    # Plot and save the plot.
    plt.plot(losses)
    plt.savefig(f'{name}.pdf')
    plt.close()

def main():

    # Accelerator instance.
    accelerator = Accelerator()

    # Get opts
    opts = get_opts()
    
    # Set random seed.
    set_seed(opts.seed)

    # Linearly scale the learning rate based on num_processes (GPUs, TPUs).
    opts.learning_rate *= accelerator.num_processes

    # Track loss
    losses = []
    
    # Begin training
    losses = train(opts, accelerator, losses)
    print("OMFG, training finished!")

    # Plot the graph
    plot_and_save(losses, opts)


if __name__ == "__main__":
    main()