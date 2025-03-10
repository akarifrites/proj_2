import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# Define transformations for the training and validation sets
transform = transforms.Compose([
    transforms.Resize((224, 224)), # 224x224 pixels
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom dataset class for test images without labels
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB to handle grayscale images
        if self.transform:
            image = self.transform(image)
        return image, img_path  # Returning img_path as an identifier


# Load the training and validation datasets
train_dataset = datasets.ImageFolder(r"C:\Users\fenel\Documents\projects\datasets\datasets\train", transform=transform)
val_dataset = datasets.ImageFolder(r"C:\Users\fenel\Documents\projects\datasets\datasets\val", transform=transform)
test_data = TestDataset(r"C:\Users\fenel\Documents\projects\datasets\datasets\test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained VGG16 model
weights = models.VGG16_Weights.DEFAULT
model = models.vgg16(weights=weights)
# Modify the classifier part of VGG16 for binary classification
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
model = model.to(device)
print(model)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 10

# Initialize lists to store metrics
train_losses = []
val_losses = []
val_accuracies = []

# Training and validation loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)

    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_losses.append(val_loss / len(test_loader))
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)
    train_losses.append(train_loss / len(train_loader.dataset))

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}, Validation Accuracy: {val_accuracy}%")
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, "
          f"Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%")
    
# Validation phase
model = model.to(device)
model.eval()

# Prepare to store filenames and predictions
image_filenames = []
predictions = []

# correct = 0
# total = 0
# Disable gradient calculation for faster prediction
with torch.no_grad():
    for images, paths in test_loader:
        images = images.to(device)
        outputs = model(images)  # Get model outputs
        _, predicted = torch.max(outputs, 1)  # Get predicted class (0 or 1)
        # total += labels.size(0)
        # correct += (predicted == labels).sum().item()

        # Map predictions to filenames and labels
        for path, pred in zip(paths, predicted):
            filename = os.path.basename(path)  # Extract filename from path
            label = 1 if pred.item() == 1 else 0  # Assign 1 for "dog", 0 for "cat"
            image_filenames.append(filename)
            predictions.append(label)

# print(f'Test Accuracy: {100 * correct / total}%')

# Save predictions to CSV
df = pd.DataFrame({'ID': image_filenames, 'Label': predictions})

# Save DataFrame to a CSV file
df.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")


# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

# Plot Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Validation Accuracy")
plt.show()


# Compute Confusion Matrix
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Generate Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()