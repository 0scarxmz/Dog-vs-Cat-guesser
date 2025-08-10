import kagglehub
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# download and unzip the dataset
path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")

# set the dataset path
dataset_path = os.path.join(path, "PetImages")

# rezise to 128x128 and transfrom the images to tensors. We some something like ([3, 128, 128])
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# load the dataset and use the PyTorch dataser class ImageFolder to create a dataset
# ImageFolder looks at subfolder names in dataset_path and it assigns labels automaticall based on aphabetical order.
# for exaple if we print(dataset.classes) we would get something like ['Cat', 'Dog'] else if we do print(dataset.class_to_idx) we would get {'Cat': 0, 'Dog': 1}
# this part is very important to understand because it is how PyTorch knows which label to assign to which image, and we will be using this later in the code
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
# create a DataLoader to load the dataset in batches
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# split the dataset into training and validation sets (80% training, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# 
# split the dataset into training and validation sets always using random_split in this type of situation to separate the dataset. Importerd in line 8
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader is being imported in line 8
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#validation loader should be shuffled to ensure that the model is evaluated on different data each time to be more accurate
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# this is the CNN model that will be used to classify the images
# You could watch this vide to understand this part better: https://youtu.be/pj9-rr1wDhM?si=TRFVtILLz4INsvOR
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool1(F.relu(self.conv3(x)))

        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
# define the model, loss function and optimizer
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

# if __name__ == "__main__": to ignore this part when working with app.py/flask
if __name__ == "__main__":

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # we are using the outputs and using .data to get the actual values of the outputs, and dim=1 to get the maximum value of the outputs
            # Calculate accuracy by comparing the predicted labels with the true labels.
            # basically if the predicted label isn't the same as the label that is being compared with, then it might return false else it returns true
            # total += labels.size(0), very understandable, then we get the accuracy by dividing the correct predictions/true by the total number of predictions
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")

