import torch
from torchvision import transforms,datasets
from torchvision.models.densenet import DenseNet161_Weights
from torch.utils.data import dataloader,DataLoader
import os
from matplotlib import pyplot as plt
import torch.nn as nn
import time

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms
from torch.multiprocessing import spawn
import torch.nn.functional as F
import argparse
import numpy as np
import csv
from sklearn.metrics import accuracy_score

# function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and test a DenseNet model on distributed GPUs.')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use for training and testing.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing.')
    args = parser.parse_args()
    return args

def initialize_csv():
    csv_file = 'training_times.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["GPUs", "Batch Size", "Training Time (Seconds)"])

# Setup DDP: Initialize Process Group
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '4444'
    # Set the random seed for reproducibility
    torch.manual_seed(42)
    # If you are working with CUDA, you might also want to set the seed for it
    torch.cuda.manual_seed(42)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Cleanup
def cleanup():
    dist.destroy_process_group()

# defining datasets
def get_dataloader(rank,world_size,batch_size,train=True):
    # Define the transformation and datasets
    augmentation_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomGrayscale(),
        transforms.RandomAffine(translate=(0.05, 0.05), degrees=0),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
])
    # Load the appropriate dataset
    if train:
        dataset = datasets.ImageFolder(root='/scratch/gohil.de/Pediatric Chest X-ray Pneumonia/train', transform=augmentation_transforms)
    else:
        dataset = datasets.ImageFolder(root='/scratch/gohil.de/Pediatric Chest X-ray Pneumonia/test', transform=test_transform)

    # Create the DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=train)

    # Create the DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return loader


class DenseNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNetModel, self).__init__()
        # Load a pre-trained DenseNet-161
        self.base_model = models.densenet161(weights=DenseNet161_Weights.DEFAULT)
        
        # Freeze the weights of the pre-trained model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Build a custom classifier
        # Note: `classifier` is a linear layer in DenseNet-161
        num_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)




# training loop
def train(rank,world_size,batch_size):
    
    setup(rank, world_size)
    print(f"Starting Training on Rank {rank}.")

    
    epoch_loss = np.array([])
    pred_list, label_list = np.array([]),np.array([])
    
    
    if rank == 0:
        start_time = time.time()  # Start timing

    
    
    # Setup training DataLoader with DistributedSampler
    train_loader = get_dataloader(rank, world_size,batch_size, train=True)
    
    # model = SimpleCNN().cuda(rank)
    model = DenseNetModel().cuda(rank)

    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.004)

    for epoch in range(5):  # just 2 epochs for demonstration
        ddp_model.train()
        t0 = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            
            inputs = inputs.cuda(rank)
            labels = labels.cuda(rank)
            
            optimizer.zero_grad()
            
            outputs = ddp_model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            
            np.append(epoch_loss, loss.cpu().data)
            _, pred = torch.max(outputs, axis=1)
            pred_list = np.append(pred_list, pred.cpu().numpy())
            label_list = np.append(label_list, labels.cpu().numpy())
            
        epoch_loss = np.asarray(epoch_loss)
        epoch_acc = accuracy_score(label_list, pred_list)    
        time_taken = time.time() - t0
       
        print(f'Rank {rank}, Epoch {epoch}, Batch {i}, Loss: {loss.item()}, Accuracy: {epoch_acc} ,Time: {time_taken}')
    
    if rank == 0:
        training_time = time.time() - start_time  # Calculate training time
        print(f"Training completed in {training_time:.2f} seconds.")
        # Write to CSV
        with open('training_times.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([world_size, batch_size, training_time])

    print(f"Training Complete on Rank {rank}.")
        
    # Only save the model from rank 0
    if rank == 0:
        # Since the model is wrapped in DDP, save the underlying model's state_dict
        torch.save(model.state_dict(), 'model_20_epoch.pth')
        print("Model saved on Rank 0.")


    
    cleanup()

    
def main():
    args = parse_arguments()

    world_size = args.num_gpus
    batch_size = args.batch_size
    initialize_csv()
    print(f"Using {world_size} GPUs with a batch size of {batch_size}.")

    # Adjust the spawn calls to pass `batch_size` as an additional argument
    spawn_args = (world_size, batch_size)
    torch.multiprocessing.spawn(train, args=spawn_args, nprocs=world_size, join=True)
    


if __name__ == "__main__":
    main()
