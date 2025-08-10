# 🐶🐱 Dog vs Cat Classifier — PyTorch + Flask

This is a complete **end-to-end machine learning project** where I trained a **Convolutional Neural Network (CNN)** in PyTorch to classify images as either **Dog** or **Cat**, and deployed it as a **Flask web app** for easy use.

I’ve included important code comments inside the repo so you can see exactly how each step works — here are some highlights.

---

## 📌 Key Code Explanations

### 1️⃣ Dataset Loading & Label Mapping
From `main.py`:
```python
# ImageFolder looks at subfolder names in dataset_path and assigns labels automatically based on alphabetical order.
# For example:
#   print(dataset.classes) -> ['Cat', 'Dog']
#   print(dataset.class_to_idx) -> {'Cat': 0, 'Dog': 1}
# This is important because later we map prediction index back to "Dog" or "Cat".
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

Splitting Train & Validation Sets
# Always use random_split here to ensure training and validation sets are randomly selected.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


CNN Architecture
From main.py:
# this is the CNN model that will be used to classify the images
# You could watch this vide to understand this part better: https://youtu.be/pj9-rr1wDhM?si=TRFVtILLz4INsvOR
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)

Accuracy Calculation
# Calculate accuracy by comparing predicted labels with true labels
_, predicted = torch.max(outputs.data, 1)
total += labels.size(0)
correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total

Flask Prediction Logic
From app.py:
# Get index of the class with the highest score
# Example: [0.2, 0.8] -> index 1 (Dog), [0.8, 0.2] -> index 0 (Cat)
_, prediction = torch.max(output, 1)
label = "Dog" if prediction.item() == 1 else "Cat"

🛠 Installation
git clone <repo-url>
cd dog-vs-cat
pip install flask torch torchvision pillow kagglehub

📊 Training
python main.py

This will:

Download the Kaggle dataset using kagglehub

Train the CNN for 10 epochs

Save weights to model.pth

🌐 Running the Web App
python app.py

then visit the page that the terminal left. 

🧠 How It Works (End-to-End)
Dataset → Downloaded & labeled automatically via folder names.

Training → CNN learns features of cats and dogs from scratch.

Saving → Model weights stored in model.pth.

Flask App → Loads weights, transforms uploaded image, predicts class.

Frontend → Displays image + predicted label.

📂 Project Structure
dog-vs-cat/
├── main.py           # Training code with detailed comments
├── app.py            # Flask web server with prediction logic
├── model.pth         # Saved model weights (after training)
├── templates/
│   └── index.html    # Upload form + results
└── static/uploads/   # Uploaded images

📜 License

---

This README both explains the project and **highlights your inline comments** so a recruiter or reviewer immediately sees that you understand the process deeply.  

Do you want me to now do the **same style README** for your **Emotion Recognition** project so they both match in format and show off your explanations? That way your portfolio will look consistent.
