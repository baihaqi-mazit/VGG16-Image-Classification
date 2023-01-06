
import os
import shutil
from datetime import datetime

import numpy as np
from pathlib import Path
import splitfolders

from sklearn.metrics import classification_report
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter, writer
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import wget
import zipfile
import torchvision.models as models
from ignite.metrics import ClassificationReport


def train(model, loader, optimizer, criterion, scheduler, saving_path, n_epochs_stop, epochs):
    epochs_no_improve = 0
    min_val_loss = np.Inf
    early_stop = False

    for i in range(1, epochs + 1):

        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_size = 0.0
            epoch_loss = 0.0
            correct = 0

            if phase == 'train':
                model.train()
            else:
                # To set dropout and batch normalization layers to evaluation mode before running inference
                model.eval()

            for images, labels in loader[phase]:

                # Move tensors to compute using gpu
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    # Predict and compute loss
                    y_pred = model(images)
                    loss = criterion(y_pred, labels)

                    # Backpropogate and update parameters
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Note: the last batch size will be the remainder of data size divided by batch size
                # Thus, we multiply one batch loss with the number of items in batch and divide it by total size later
                # The final running size is equal to the data size
                running_loss += loss.item() * y_pred.size(0)
                running_size += y_pred.size(0)

                # The predictions is the index of the maximum values in the output of model
                predictions = torch.max(y_pred, 1)[1]

                correct += (predictions == labels).sum().item()

            epoch_loss = running_loss / running_size
            epoch_accuracy = correct / running_size
            writer.add_scalars('Loss', {phase: epoch_loss}, i)
            writer.add_scalars('Accuracy', {phase: epoch_accuracy}, i)

            # Print score at every epoch
            if (i % 1 == 0 or i == 1):
                if phase == 'train':
                    scheduler.step(epoch_loss)
                    print(f'Epoch {i}:')
                print(f'  {phase.upper()} Loss: {epoch_loss}')
                print(f'  {phase.upper()} Accuracy: {epoch_accuracy}')

            # For visualization (Optional)
            loss_score[phase].append(epoch_loss)

            # Early stopping
            if phase == 'valid':
                if epoch_loss < min_val_loss:
                    # Save the model before the epochs start to not improving
                    torch.save(model.state_dict(), saving_path)
                    print(f"Model saved at Epoch {i} \n")
                    epochs_no_improve = 0
                    min_val_loss = epoch_loss

                else:
                    epochs_no_improve += 1
                    print('\tepochs_no_improve:',
                          epochs_no_improve, ' at Epoch', i)

            if epochs_no_improve == n_epochs_stop:
                print('\nEarly stopping!')
                early_stop = True
                break

        # To exit loop
        if early_stop:
            print("Stopped")
            break

    writer.close()


if __name__ == "__main__":

    if os.path.exists("resources/data/train") or os.path.exists("resources/data/val"):
        shutil.rmtree("resources/data/train")
        shutil.rmtree("resources/data/val")

    splitfolders.ratio('resources/data/all', output="resources/data", seed=1337, ratio=(.8, 0.2))
    out_file = open("label.txt",'w')

    # The data is located in the resources/data folder
    datadir = 'resources/data'
    traindir = datadir + '/train/'
    validdir = datadir + '/val/'

    # Check our images number in the train, val and test folders (Optional)
    # Iterate through each category
    categories = []
    train_size, val_size = 0, 0

    for category in os.listdir(traindir):
        if category != ".DS_Store":
            categories.append(category)

            # Number of images added up
            train_imgs = os.listdir(Path(traindir) / f'{category}')
            valid_imgs = os.listdir(Path(validdir) / f'{category}')
            train_size += len(train_imgs)
            val_size += len(valid_imgs)

    print(f'Train set: {train_size}, Validation set: {val_size}', end='\n\n')
    print(categories)
    print(f'\nNumber of categories: {len(categories)}')

    # We will need our input in tensors form, must `transforms.ToTensor(),`

    image_transforms = {
        # Train uses data augmentation
        'train':
            transforms.Compose([
                # You can set 'Resize' and 'Crop' to higher resolution for better result
                transforms.Resize((500,500)),
                # transforms.Grayscale(3),

                # Data augmented here
                # Use (224, 244) if you want to train on Imagenet pre-trained model
                transforms.RandomCrop(500),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),

                transforms.ToTensor(),
            ]),

        # Validation and Inference do not use augmentation
        'valid':
            transforms.Compose([
                # You can set to higher resolution for better result
                transforms.Resize((500,500)),
                # transforms.Grayscale(3),
                # transforms.CenterCrop(150),
                transforms.ToTensor(),
            ]),
    }

    torch.manual_seed(123)

    # Datasets from folders
    data = {
        'train':
            datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
        'valid':
            datasets.ImageFolder(root=validdir, transform=image_transforms['valid']),
    }

    class_names = data['train'].classes
    print(class_names)
    for x in class_names:
        out_file.write(x)
        out_file.write('\n')
    out_file.close()
    # Dataloader iterators, make sure to shuffle
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=10, shuffle=True),
        'valid': DataLoader(data['valid'], batch_size=10, shuffle=True)
    }

    # To check the iterative behavior of the DataLoader (optional)
    # Iterate through the dataloader once
    trainiter = iter(dataloaders['train'])
    features, labels = next(trainiter)
    print(features.shape)
    print(labels.shape)

    # Visualization of images in dataloader (optional)
    # nrow = Number of images displayed in each row of the grid.
    # Clip image pixel value to 0-1
    grid = torchvision.utils.make_grid(np.clip(features[0:10], 0, 1), nrow=10)

    plt.figure(figsize=(15, 15))
    # Transpose to show in rows / horizontally
    plt.imshow(np.transpose(grid, (1, 2, 0)))

    print("Labels: ")
    print(labels[0:10])
    for i in labels[0:10]:
        print(class_names[i] + ", ", end="")

    # Define model, dataloaders, optimizer, criterion
    model = models.vgg16(pretrained=True)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    criterion = nn.CrossEntropyLoss()

    # Freeze pre-trained model weights
    # for param in model.parameters():
    #     param.requires_grad = False

    # Add custom classifier / output layer
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 256),  # model.classifier[6].in_features = 4096
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, len(class_names)),
        nn.LogSoftmax(dim=1))

    print(f'Fully connected layers in the pre-trained model:\n {model.classifier}')

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    # Move model and variables to gpu torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    # Setup writer for tensorboard and also a saving_path to save model (Optional)
    writer = SummaryWriter('logs/glare_classifier')
    if not os.path.exists('resources/model'):
        os.mkdir('resources/model')
    combine_date_time = datetime.now().strftime("%Y%m%d%H%M%S")
    saving_path = f'resources/model/glare_classifier_state_dict{combine_date_time}.pt'

    n_epochs_stop = 50
    epochs = 100
    loss_score = {'train': [], 'valid': []}

    train(model, dataloaders, optimizer, criterion, scheduler,
          saving_path, n_epochs_stop, epochs)

    fig, ax = plt.subplots()
    fig.set_size_inches(14, 7)
    ax.set_title("Loss Score against Epoch")
    ax.grid(visible=True)
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel("Loss Score")
    ax.plot(loss_score['train'], color='red', label='Training Loss')
    ax.plot(loss_score['valid'], color='green', label='Validation Loss')
    ax.legend()
    plt.show()
