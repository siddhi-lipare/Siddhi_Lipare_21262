import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define transforms for data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load SVHN dataset
train_dataset = SVHN(root='./data', split='train', download=True, transform=transform)
test_dataset = SVHN(root='./data', split='test', download=True, transform=transform)

# Use DataLoader for batching and shuffling data
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load pretrained ResNet-50 model
model = torchvision.models.resnet50(pretrained=True)

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Modify the last fully connected layer for the number of classes in SVHN dataset
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
model.train()
for epoch in range(5):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

# Evaluate the model
model.eval()
preds = []
targets = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        preds.extend(predicted.tolist())
        targets.extend(labels.tolist())

# Calculate performance metrics
accuracy = accuracy_score(targets, preds)
precision = precision_score(targets, preds, average='macro')
recall = recall_score(targets, preds, average='macro')
f1 = f1_score(targets, preds, average='macro')

print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
