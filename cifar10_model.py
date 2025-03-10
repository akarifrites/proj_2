import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import os


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# Define a function to load all CIFAR-10 batches and combine them
def load_cifar10_data(data_dir):
    # Load each data batch
    images = []
    labels = []
    for i in range(1, 6):  # CIFAR-10 has 5 training batches
        batch = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
        images.append(batch[b'data'])
        labels += batch[b'labels']

    # Stack and reshape images, and convert labels to a numpy array
    images = np.vstack(images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Shape: (50000, 32, 32, 3)
    labels = np.array(labels)
    
    # Load the test batch
    test_batch = unpickle(os.path.join(data_dir, 'test_batch'))
    test_images = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_labels = np.array(test_batch[b'labels'])
    
    return images, labels, test_images, test_labels


# Define a custom Dataset class for PyTorch
class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image)  # Convert numpy array to PIL image
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    # change data_dir to the path of your CIFAR-10 dataset
    data_dir = r"C:\Users\fenel\Documents\projects\cifar-10-python\cifar-10-batches-py"
    train_images, train_labels, test_images, test_labels = load_cifar10_data(data_dir)

    # Data transformations and loaders
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    # Create Dataset and DataLoader for training and testing
    train_dataset = CIFAR10Dataset(train_images, train_labels, transform=transform_train)
    test_dataset = CIFAR10Dataset(test_images, test_labels, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)


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


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a pretrained ResNet model and modify the classifier layer
    weights = models.ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=weights)
    resnet.fc = nn.Sequential(
        nn.Linear(resnet.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 10)  # Updated for 10 classes
    )
    resnet = resnet.to(device)


    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
    num_epochs = 10

    # List to store test accuracy after each epoch
    test_accuracies = []

    # Train ResNet 
    for epoch in range(num_epochs):
        resnet.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)
            
            # Forward pass
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # Evaluate on test set
        resnet.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = resnet(features)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)  # Store accuracy for plotting
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss/len(train_loader)}, Test Accuracy: {accuracy:.2f}%")

      
    resnet.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = resnet(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
  

    # Plot test accuracy over epochs
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), test_accuracies, marker='o')
    plt.title("Test Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.grid()
    plt.show()