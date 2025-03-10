import torch
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define transformations for data augmentation
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

transform_val_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load datasets
train_data = datasets.ImageFolder(r"C:\Users\fenel\Documents\AIDM_CatsDogs\datasets\datasets\train", transform=transform_train)
val_data = datasets.ImageFolder(r"C:\Users\fenel\Documents\AIDM_CatsDogs\datasets\datasets\val", transform=transform_val_test)
test_data = datasets.ImageFolder(r"C:\Users\fenel\Documents\AIDM_CatsDogs\datasets\datasets\test", transform=transform_val_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)



# Load a pretrained ResNet model and modify the classifier layer
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 2)  # Binary output for 'dog' and 'cat'
)
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

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

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, "
          f"Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%")


model.eval()
image_ids = []
predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        image_ids.extend([path.split('/')[-1].split('.')[0] for path, _ in test_loader.dataset.samples])
        predictions.extend(predicted.cpu().numpy())

# Save predictions to CSV
submission = pd.DataFrame({'ID': image_ids, 'Label': predictions})
submission['Label'] = submission['Label'].map({1: 'dog', 0: 'cat'})
submission.to_csv('submission.csv', index=False)