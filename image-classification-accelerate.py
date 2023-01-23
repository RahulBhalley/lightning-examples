import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, FakeData, CelebA
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from accelerate import Accelerator
# from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import os
from multiprocessing import cpu_count
from argparse import ArgumentParser

print(f"PyTorch version: {torch.__version__}")
print(f"CPU cores: {cpu_count()}")


def get_device(accelerator):
    return accelerator.device

def get_model(opts):
    if opts.network == "resnet34":
        return models.resnet34(False)
    elif opts.network == "resnet50":
        return models.resnet50(False)
    elif opts.network == "resnet152":
        return models.resnet152(False)
    elif opts.network == "alexnet":
        return models.alexnet(False)
    elif opts.network == "mobilenet_v3_small":
        return models.mobilenet_v3_small(False)
    elif opts.network == "efficientnet_b0":
        return models.efficientnet_b0()

def get_optimizer(opts, model):
    if opts.optim_name == "adam":
            return optim.Adam(model.parameters())

def get_dataloader(opts):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((opts.image_size, opts.image_size)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = CIFAR10(root='.', train=True,  download=True, transform=transform)
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

def train(opts, accelerator):

    # get everything!
    device = get_device(accelerator)
    dataloader = get_dataloader(opts)
    model = get_model(opts)
    # loss_fn = get_loss_fn(opts)
    optimizer = get_optimizer(opts, model)
    
    # Prepare for Accelerate
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Move model to device
    model.to(device)

    # Start training
    loss = 0.0
    for epoch in range(opts.max_epochs):
        for batch in tqdm(dataloader, desc=f"[Epoch: {epoch} | Loss: {loss}]"):
            optimizer.zero_grad()
            inputs, targets = batch
            preds = model(inputs)
            loss = F.cross_entropy(preds, targets)
            accelerator.backward(loss)
            optimizer.step()

def get_opts():
    parser = ArgumentParser()
    parser.add_argument("--network", default="resnet50", type=str)
    parser.add_argument("--max_epochs", default=50, type=int)
    parser.add_argument('--accelerator', default='auto', type=str, help='Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”) as well as custom accelerator instances')
    parser.add_argument('--devices', default=None, type=int, help='Will be mapped to either gpus, tpu_cores, num_processes or ipus, based on the accelerator type')
    parser.add_argument('--strategy', default=None, type=str, help='Strategy controls the model distribution across training, evaluation, and prediction to be used by the Trainer')
    parser.add_argument('--precision', default=32, type=int, help='Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16). Can be used on CPU, GPU, TPUs, HPUs or IPUs. Default: 32')
    parser.add_argument('--accumulate_grad_batches', default=None, type=int, help='Accumulates grads every k batches or as set up in the dict. Default: None')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
    parser.add_argument("--image_size", default=32, type=int, help="The image size of dataset.")
    parser.add_argument('--workers', default=cpu_count(), type=int, help='Number of train dataloader workers')
    parser.add_argument('--persistent_workers', default=False, action="store_true")
    parser.add_argument('--learning_rate', default=0.5, type=float, help='Optimizer learning rate')
    parser.add_argument('--optim_name', default='adam', type=str, help='Which optimizer to use')
    parser.add_argument("--reload_dataloaders_every_n_epochs", default=1, type=int, help="Reloads dataloaders after what epoch?")
    opts = parser.parse_args()
    return opts

if __name__ == "__main__":

    accelerator = Accelerator()

    # Get opts
    opts = get_opts()
    
    # Begin training
    train(opts, accelerator)

    print("OMFN, training finished!")
