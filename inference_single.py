import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models

# To load a saved state dict, we need to instantiate the model first
best_model = models.vgg16(pretrained=True)

# Add custom classifier / output layer
best_model.classifier[6] = nn.Sequential(
    nn.Linear(4096, 256),  # model.classifier[6].in_features = 4096
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2),
    nn.LogSoftmax(dim=1))

# Notice that the load_state_dict() function takes a dictionary object, NOT a path to a saved object.
# This means that you must deserialize the saved state_dict before you pass it to the load_state_dict() function.
best_model.load_state_dict(torch.load(
    'resources/model/glare_classifier_state_dict20220707202709.pt'))
best_model.eval()


def image_loader(image_name):
    image = Image.open(image_name).convert("RGB")
    image = loader(image).float()
    image = Variable(image)
    image = image.unsqueeze(0)
    return image

img_path = 'non-glare/clear30.jpg'

loader = transforms.Compose([transforms.Resize(500), transforms.ToTensor()])
class_names = []
my_class = {}

with open("label.txt") as f:
    content = f.readlines()
    class_names = [x.strip().lower() for x in content]

for x, category in enumerate(class_names):
    my_class[class_names[x]] = x

print(my_class)

with torch.no_grad():
    image = image_loader(img_path)
    y_pred = best_model(image)
    a = torch.max(y_pred, 1)[1].tolist()
    a = a[0]
    class_glare = [new_k for new_k in my_class.items() if new_k[1] == a]
    print(class_glare)
