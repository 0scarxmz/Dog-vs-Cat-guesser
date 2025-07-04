from flask import Flask, render_template, request
import torch
from torchvision import transforms
import os
from main import CNNModel
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'


model = CNNModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        img = request.files['image']
        img_path = os.path.join(UPLOAD_FOLDER, img.filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        img.save(img_path)

        img_raw = Image.open(img_path)
        image = transform(img_raw).unsqueeze(0)

        output = model(image)
        _, prediction = torch.max(output, 1)
        label = "Dog" if prediction.item() == 1 else "Cat"

        return render_template('index.html', prediction=label, image_path=img_path)
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)