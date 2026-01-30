import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def create_resnet_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 100) # CIFAR-100
    return model.to(device)

def train_model(model, loader, epochs=2):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

def evaluate_benchmarks(model, loader):
    model.eval()
    correct, total = 0, 0
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024**2)
    
    start_time = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    
    latency = (time.time() - start_time) / total * 1000 # ms/image
    mem_after = process.memory_info().rss / (1024**2)
    return (100 * correct / total), latency, max(0, mem_after - mem_before)


def visualize_comparison(dataset_normal, dataset_fovea, n=3):
    fig, axes = plt.subplots(2, n, figsize=(10, 6))
    for i in range(n):
        # Original (Normal)
        img_n, _ = dataset_normal[i]
        img_n = img_n.permute(1, 2, 0).numpy()
        img_n = (img_n - img_n.min()) / (img_n.max() - img_n.min()) # Normalize for display
        axes[0, i].imshow(img_n)
        axes[0, i].set_title("Original RGB")
        axes[0, i].axis("off")

        # Foveated
        img_f, _ = dataset_fovea[i]
        img_f = img_f.permute(1, 2, 0).numpy()
        img_f = (img_f - img_f.min()) / (img_f.max() - img_f.min())
        axes[1, i].imshow(img_f)
        axes[1, i].set_title("Foveated RGB")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.show()