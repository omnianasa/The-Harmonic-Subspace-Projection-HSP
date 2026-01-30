from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from foveated import FoveatedTransform
from utils import create_resnet_model, train_model, evaluate_benchmarks, visualize_comparison
from seed import set_seed

SEEDS = [0, 42, 123, 999, 1024] 
stats = {"Normal": [], "Foveated": []}

transform_base = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
transform_fovea = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), FoveatedTransform()])

for mode in ["Normal", "Foveated"]:
    print(f"\nEvaluating Mode: {mode}")
    t = transform_base if mode == "Normal" else transform_fovea
    train_ds = datasets.CIFAR100("./data", train=True, download=True, transform=t)
    test_ds = datasets.CIFAR100("./data", train=False, download=True, transform=t)
    
    if mode == "Foveated": visualize_comparison(datasets.CIFAR100("./data", train=False, transform=transform_base), test_ds)

    loader_train = DataLoader(train_ds, batch_size=64, shuffle=True)
    loader_test = DataLoader(test_ds, batch_size=64, shuffle=False)

    for seed in SEEDS:
        set_seed(seed)
        model = create_resnet_model()
        train_model(model, loader_train, epochs=2)
        acc, lat, mem = evaluate_benchmarks(model, loader_test)
        stats[mode].append([acc, lat, mem])


print("\n" + "="*50)
print(f"{'Metric':<20} | {'Normal (Mean)':<15} | {'Foveated (Mean)':<15}")
print("-" * 50)
metrics = ["Accuracy (%)", "Latency (ms/img)", "Memory (MB)"]
for i, name in enumerate(metrics):
    m_norm = np.mean([s[i] for s in stats["Normal"]])
    m_fovea = np.mean([s[i] for s in stats["Foveated"]])
    print(f"{name:<20} | {m_norm:<15.4f} | {m_fovea:<15.4f}")
print("="*50)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['#4A90E2', '#50E3C2']
for i, name in enumerate(metrics):
    vals = [np.mean([s[i] for s in stats["Normal"]]), np.mean([s[i] for s in stats["Foveated"]])]
    axes[i].bar(["Normal", "Foveated"], vals, color=colors)
    axes[i].set_title(name)
plt.show()