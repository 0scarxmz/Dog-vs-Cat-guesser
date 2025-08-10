from flask import Flask, render_template, request
import torch
from torchvision import transforms
import os
from main import CNNModel # impoets the CNNModel class from main.py
from PIL import Image

app = Flask(__name__) # Flask app initialization
UPLOAD_FOLDER = 'static/uploads' # folder to save uploaded images

model = CNNModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))

# already loaded the model, so we can set it to evaluation mode
model.eval()

# define the image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# route for the home page
@app.route('/', methods=['GET', 'POST'])

def index():
    # request.method is for the image upload
    if request.method == 'POST':
        img = request.files['image']
        # pasting the image to the upload folder
        img_path = os.path.join(UPLOAD_FOLDER, img.filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        img.save(img_path)

        # open the image, apply the transformation and add a batch dimension
        img_raw = Image.open(img_path)
        image = transform(img_raw).unsqueeze(0)

        # make the prediction
        output = model(image)
        _, prediction = torch.max(output, 1) # get the index of the class with the highest score. Using dim=1 to get the index of the class with the highest score. That is where we get [0.2, 0.8(highest number) cat] or [0.8(cat), 0.2] for example.
        # If the output is [0.2, 0.8], prediction will be 1/dot else if the output is [0.8, 0.2], prediction will be 0/cat. 
        label = "Dog" if prediction.item() == 1 else "Cat"

        # connecting the prediction and the image path to the HTML template
        return render_template('index.html', prediction=label, image_path=img_path)
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    # running the Flask app in debug mode
    app.run(debug=True)