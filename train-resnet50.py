import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Normalize, Resize
from torchgeo.datasets import EuroSAT
import matplotlib.pyplot as plt

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

eurosat_path = "data"

# Define class mapping
CLASS_MAPPING = {
    "AnnualCrop": 0,
    "Forest": 1,
    "HerbaceousVegetation": 2,
    "Highway": 3,
    "Industrial": 4,
    "Pasture": 5,
    "PermanentCrop": 6,
    "Residential": 7,
    "River": 8,
    "SeaLake": 9
}

# Custom Transform Wrapper
class ApplyToImageAndLabel:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        sample["image"] = self.transform(sample["image"])  # Apply transforms to the "image"
        if isinstance(sample["label"], str):  # Convert string label to integer
            sample["label"] = CLASS_MAPPING[sample["label"]]
        return sample

# Custom collate function for DataLoader
def custom_collate_fn(batch):
    images = torch.stack([sample["image"] for sample in batch])
    labels = torch.tensor([sample["label"] for sample in batch])
    return images, labels

# Define transformations
transform = ApplyToImageAndLabel(Compose([
    Resize((224, 224)),  # Resize images for ResNet50
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for ImageNet-trained models
]))

# Load EuroSAT Dataset for training, validation, and testing
train_dataset = EuroSAT(root=eurosat_path, split="train", bands=["B04", "B03", "B02"], transforms=transform, download=False)
val_dataset = EuroSAT(root=eurosat_path, split="val", bands=["B04", "B03", "B02"], transforms=transform, download=False)
test_dataset = EuroSAT(root=eurosat_path, split="test", bands=["B04", "B03", "B02"], transforms=transform, download=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

# Define the Model (ResNet50)
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Use updated weights argument

# Define a function to insert Dropout in a Sequential block
def add_dropout_to_sequential(module, p=0.2):
    new_module = nn.Sequential()
    for name, layer in module.named_children():
        new_module.add_module(name, layer)
        # Add Dropout after ReLU activations
        if isinstance(layer, nn.ReLU):
            new_module.add_module(f"dropout_{name}", nn.Dropout(p=p))
    return new_module

# Apply dropout to layer4 (or any other layer you choose)
model.layer4 = add_dropout_to_sequential(model.layer4, p=0.3)


model.fc = nn.Linear(model.fc.in_features, 10)  # EuroSAT has 10 classes

print(model)

model = model.to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop with Validation Checkpointing
epochs = 10
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

best_val_accuracy = 0.0
best_model_path = "resnet50_eurosat_best.pth"

for epoch in range(epochs):
    # Training phase
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_losses.append(total_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)

    # Validation phase
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_losses.append(val_loss / len(val_loader))
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, "
          f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}%")

    # Checkpoint the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with Val Accuracy: {best_val_accuracy:.2f}%")

# Save the final model
final_model_path = "resnet50_eurosat_final.pth"
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to '{final_model_path}'.")



# Test Evaluation
model.load_state_dict(torch.load(best_model_path))  # Load the best model for testing
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy (Best Model): {100 * correct / total:.2f}%")

# Plot Training and Validation Loss/Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss vs Epochs")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy vs Epochs")
plt.show()