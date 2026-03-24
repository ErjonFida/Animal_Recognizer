import os
import scipy.io
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import ImageFolder

# Define transform
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])


class SimpleCnn(nn.Module):
    def __init__(self):
        super(SimpleCnn, self).__init__()

        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.Conv2d(out_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.features = nn.Sequential(
            conv_block(3, 32),    # 224 -> 112
            conv_block(32, 64),   # 112 -> 56
            conv_block(64, 128),  # 56 -> 28
            conv_block(128, 256), # 28 -> 14
            conv_block(256, 512)  # 14 -> 7
        )

        # Global Average Pooling reduces spatial dimensions to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 15)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

def train_epoch(model, train_loader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device, dtype=torch.long)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Track progress
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Print progress
        if batch_idx % 20 == 0 and batch_idx > 0:
            avg_loss = running_loss / 20
            accuracy = 100. * correct / total
            processed_samples = batch_idx * len(data)
            total_samples = len(train_loader.dataset)
            print(f' [{processed_samples} / {total_samples}] '
                  f'Loss: {avg_loss:.3f} | Accuracy: {accuracy:.1f}%')
            running_loss = 0.0


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy
train_dataset = ImageFolder('Kaggle_Animal_data/Training Data/Training Data', transform=train_transform)
val_dataset   = ImageFolder('Kaggle_Animal_data/Validation Data/Validation Data', transform=val_transform)
test_dataset  = ImageFolder('Kaggle_Animal_data/Testing Data/Testing Data', transform=val_transform)

print(f"Loaded {len(train_dataset)} training images from {len(train_dataset.classes)} classes.")
print(f"Loaded {len(val_dataset)} validation images from {len(val_dataset.classes)} classes.")
print(f"Loaded {len(test_dataset)} testing images from {len(test_dataset.classes)} classes.")

# For training shuffle = True
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# For testing and validation
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#Get one batch from each

for name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
    images, labels = next(iter(loader))
    print(f"{name} batch: {images.shape}")

# Get one batch from each (safely unpack since they might throw an error if empty)

try:
    import torch_directml
    device = torch_directml.device()
    print("Using Intel GPU via DirectML!")
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
model = SimpleCnn().to(device)

# Initialize weights with parameters from the last epoch
start_epoch = 1
saved_models = [f for f in os.listdir('.') if f.startswith('Animal_model_epoch_') and f.endswith('.pth')]
if saved_models:
    latest_epoch = max([int(f.split('_')[-1].split('.')[0]) for f in saved_models])
    latest_model_path = f"Animal_model_epoch_{latest_epoch}.pth"
    model.load_state_dict(torch.load(latest_model_path, map_location=device, weights_only=False))
    print(f"Initialized weights with parameters from {latest_model_path}")
    start_epoch = latest_epoch + 1

# Loss function and optimizer (added weight decay to help generalization)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4, foreach=False)

if start_epoch > 1:
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = 0.0001

# Cosine annealing scheduler for smoother LR decay
epochs = 5
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6, last_epoch=(start_epoch - 1) if start_epoch > 1 else -1)



if __name__ == "__main__":
    # epochs is already set above with the scheduler
    
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch} | Current LR: {scheduler.get_last_lr()[0]:.6f}")
        train_epoch(model, train_loader, loss_function, optimizer, device)
        
        val_acc = evaluate(model, val_loader, device)
        print(f'Validation Accuracy: {val_acc:.2f}%')
        
        # Step the scheduler
        scheduler.step()
        
        # Save model parameters after each epoch
        save_path = f"Animal_model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model parameters saved to {save_path}")

# Final evaluation on test set
print("\n" + "="*30)
print("Final Evaluation on Test Set:")
test_acc = evaluate(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.2f}%")
print("="*30)
