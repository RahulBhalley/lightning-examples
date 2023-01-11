import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, FakeData, CelebA
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
# from efficientnet_pytorch import EfficientNet
import pytorch_lightning as pl
from tqdm import tqdm
import os
from multiprocessing import cpu_count
from argparse import ArgumentParser

print(f"PyTorch version: {torch.__version__}")
print(f"Lighting version: {pl.__version__}")
print(f"CPU cores: {cpu_count()}")

class ImageClassificationAlgorithm(pl.LightningModule):

    def __init__(self, opts):
        super().__init__()

        if opts.network == "resnet34":
            self.network = models.resnet34(False)
        elif opts.network == "resnet50":
            self.network = models.resnet50(False)
        elif opts.network == "resnet152":
            self.network = models.resnet152(False)
        elif opts.network == "alexnet":
            self.network = models.alexnet(False)
        elif opts.network == "mobilenet_v3_small":
            self.network = models.mobilenet_v3_small(False)
        elif opts.network == "efficientnet_b0":
            self.network = models.efficientnet_b0()
        
        self.opts = opts

    def forward(self, inputs):
        return self.network(inputs)

    def training_step(self, batch, batch_idx):
        # A single training step of training loop.
        # print(f"batch size: {self.batch_size}")
        inputs, targets = batch
        preds = self.network(inputs)
        loss = F.cross_entropy(preds, targets)
        return loss

    def configure_optimizers(self):
        if self.opts.optim_name == "adam":
            return optim.Adam(self.parameters())

def get_opts():
    parser = ArgumentParser()
    parser.add_argument("--network", default="resnet50", type=str)
    parser.add_argument("--max_epochs", default=2, type=int)
    parser.add_argument('--accelerator', default='auto', type=str, help='Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”) as well as custom accelerator instances')
    parser.add_argument('--devices', default=None, type=int, help='Will be mapped to either gpus, tpu_cores, num_processes or ipus, based on the accelerator type')
    parser.add_argument('--strategy', default=None, type=str, help='Strategy controls the model distribution across training, evaluation, and prediction to be used by the Trainer')
    parser.add_argument('--precision', default=32, type=int, help='Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16). Can be used on CPU, GPU, TPUs, HPUs or IPUs. Default: 32')
    parser.add_argument('--accumulate_grad_batches', default=None, type=int, help='Accumulates grads every k batches or as set up in the dict. Default: None')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
    parser.add_argument("--image_size", default=32, type=int, help="The image size of dataset.")
    parser.add_argument('--workers', default=2, type=int, help='Number of train dataloader workers')
    parser.add_argument('--persistent_workers', default=False, action="store_true")
    parser.add_argument('--learning_rate', default=0.5, type=float, help='Optimizer learning rate')
    parser.add_argument('--optim_name', default='adam', type=str, help='Which optimizer to use')
    opts = parser.parse_args()
    return opts

if __name__ == "__main__":

    # Get opts
    opts = get_opts()
    
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

    model = ImageClassificationAlgorithm(opts)
    trainer = pl.Trainer(
        max_steps=opts.max_steps,
        accumulate_grad_batches=opts.accumulate_grad_batches,
        precision=opts.precision,
        accelerator=opts.accelerator,
        devices=opts.devices,
        strategy=opts.strategy
    )
    trainer.fit(model, trainloader)
    