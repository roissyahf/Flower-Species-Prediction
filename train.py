# This script will train a new network on a dataset and save the model as checkpoint

# Import required libraries
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

# argparse object
# Create Parse using ArgumentParser
parser = argparse.ArgumentParser(description='train model for classifying flowers')
# argument 1: it's a path to the dataset folder
parser.add_argument('--dir', type = str, default = '/flowers', 
                    help = 'path to the folder of flowers images')
# argument 2: set directory to save model checkpoints
parser.add_argument('--save_dir', type = str, default = '/model_checkpoints',
                    help = 'path to the folder of model checkpoints')
# argument 3: it's an architecture of the pretrained model 
parser.add_argument('--arch', type = str, default = 'vgg13', choices = ['vgg11', 'vgg13', 'vgg16'], help = 'architecture of the pretrained model')
# argument 4: it's a learning_rate
parser.add_argument('--learning_rate', type = float, default = 0.001,
                    help = 'learning rate for model training')
# argument 5: it's a hidden_units
parser.add_argument('--hidden_units', type = int, default = 128,
                    help = 'hidden layers for model training')
# argument 6: it's an epochs
parser.add_argument('--epochs', type = int, default = 5,
                    help = 'total epochs for model training')
# argument 7: it's for GPU usage
parser.add_argument('--gpu', action = "store_true", default = True,
                    help = 'use GPU for model training')
args_train = parser.parse_args()

# Define training, and validation sets transforms
data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),

    'valid': transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
}

# Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(root=args_train.dir + '/' + x, transform = data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'valid']}

# Check if GPU is available and if the user wants to use it
device = torch.device("cuda:0" if args_train.gpu and torch.cuda.is_available() else "cpu")

# Use the specified architecture
if args_train.arch == 'vgg11':
    model = models.vgg11(pretrained=True)
    input_size = model.classifier[0].in_features  # VGG11 input size: 25088
elif args_train.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
    input_size = model.classifier[0].in_features  # VGG13 input size: 25088
elif args_train.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    input_size = model.classifier[0].in_features  # VGG16 input size: 25088
else:
    raise ValueError('Unsupported architecture. Supported architectures are vgg11, vgg13, and vgg16.')

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Define the classifier
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, args_train.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(args_train.hidden_units, 102)),
                          ('dropout', nn.Dropout(0.25)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

# Define the criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args_train.learning_rate)

# Move model to the default device
model.to(device)

# Training loop
steps = 0
print_every = 5
for epoch in range(args_train.epochs):
    training_loss = 0
    for inputs, labels in dataloaders['train']:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)

        logps = model.forward(inputs)
        loss = criterion(logps, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        if steps % print_every == 0:
            validation_loss = 0
            validation_acc = 0
            model.eval()

            with torch.no_grad():
                for inputs, labels in dataloaders['valid']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    validation_loss += batch_loss.item()

                    # calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    validation_acc += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{args_train.epochs}.. "
                  f"Training loss: {training_loss/print_every:.3f}.. "
                  f"Validation loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
                  f"Validation accuracy: {validation_acc/len(dataloaders['valid']):.3f}")
            
            training_loss = 0
            model.train()

# Save the checkpoints
checkpoint = {'architecture': args_train.arch,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': image_datasets['train'].class_to_idx}

torch.save(checkpoint, args_train.save_dir + '/checkpoints.pth')
print(f"Checkpoint saved to {args_train.save_dir}/checkpoints.pth")