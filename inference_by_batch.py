import torch
import torch.nn as nn
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

# The data is located in the resources/data folder
datadir = 'resources/data/'
testdir = datadir + '/train/'
dirtytestdir = datadir + '/dirty_test/'
testdir2 = datadir + '/test/'

image_transforms = {
    # Train uses data augmentation
    'train':
        transforms.Compose([
            # You can set 'Resize' and 'Crop' to higher resolution for better result
            transforms.Resize(180),

            # Data augmented here
            # Use (224, 244) if you want to train on Imagenet pre-trained model
            transforms.RandomCrop(150),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
        ]),

    # Validation and Inference do not use augmentation
    'valid':
        transforms.Compose([
            # You can set to higher resolution for better result
            transforms.Resize(500),
            # transforms.CenterCrop(256),
            # transforms.Grayscale(3),
            transforms.ToTensor(),
        ]),
}

test_data = datasets.ImageFolder(
    root=dirtytestdir, transform=image_transforms['valid'])
testloader = DataLoader(test_data, len(test_data), shuffle=False)

with torch.no_grad():
    for images, labels in testloader:
        correct = 0
        imagess = images
        y_pred = best_model(images)
        predictions = torch.max(y_pred, 1)[1]
        print(predictions)
        print(labels)
        correct += (predictions == labels).sum().item()
        accuracy = correct / len(test_data)
        print(f"Test Accuracy: {accuracy}")