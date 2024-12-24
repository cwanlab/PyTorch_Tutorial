import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

# Model
input_size = 28*28
hidden_size = 100
num_classes = 10
device = torch.device("cuda")

FILE = "outputs/mnist_ffn.pth"

class MultiCLSNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiCLSNeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

model = MultiCLSNeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
model.load_state_dict(torch.load(FILE, map_location=device))
model.to(device)
model.eval()

# image --> tensor
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0).to(device)

# prediction
def get_prediction(image_tensor):
    images = image_tensor.reshape(-1, 28*28)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

with open("data/7.png", "rb") as f:
    image_bytes = f.read()
tensor = transform_image(image_bytes)
prediction = get_prediction(tensor)
print("prediction:", prediction)