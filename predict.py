# This script will predict the class probability of a single image

# Import required libraries
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json

# Create Parse using ArgumentParser
parser = argparse.ArgumentParser()
# argument 1: single image to be predicted
parser.add_argument('--input_dir', type = str, 
                    help = 'path to the single image for prediction')
# argument 2: model chekpoint path
parser.add_argument('--checkpoint_dir', type = str,
                    help = 'path to the model checkpoint')
# argument 3: return top K most likely classes
parser.add_argument('--top_k', default = 3, type = int,
                    help = 'top K most likely classes')
# argument 4: it's a JSON file to mapping categories
parser.add_argument('--category_names', default = 'cat_to_name.json',
                    help = 'use a mapping of categories to real names')
# argument 5: use GPU for inference
parser.add_argument('--gpu', action = 'store_true', default = True,
                    help = 'GPU usage for model training')
args_predict = parser.parse_args()

# Loads a checkpoint and rebuilds the model
checkpoint = torch.load(args_predict.checkpoint_dir)

# Use the specified architecture
if checkpoint['architecture'] == 'vgg11':
    model = models.vgg11(pretrained=True)
    input_size = model.classifier[0].in_features  # VGG11 input size: 25088
elif checkpoint['architecture'] == 'vgg13':
    model = models.vgg13(pretrained=True)
    input_size = model.classifier[0].in_features  # VGG13 input size: 25088
elif checkpoint['architecture'] == 'vgg16':
    model = models.vgg16(pretrained=True)
    input_size = model.classifier[0].in_features  # VGG16 input size: 25088
else:
    raise ValueError('Unsupported architecture. Supported architectures are vgg11, vgg13, and vgg16.')

# Replace the classifier with the one saved in checkpoint
model.classifier = checkpoint['classifier']

# Load the state dict into the model
model.load_state_dict(checkpoint['state_dict'])

# Process image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # resize the images to 224x22
    img_size = (224,224)
    image = image.resize(img_size)

    # convert integers 0-255 to floats 0-1
    np_image = np.array(image) / 255.0

    # normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # reorder dimension
    np_image = np_image.transpose((2,0,1))

    # return a numpy array
    return np_image                           

# Label mapping
with open(args_predict.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Load the image from image_path
image = Image.open(args_predict.input_dir)

# Preprocess image
np_image = process_image(image)

# Convert numpy array to torch tensor
tensor_image = torch.from_numpy(np_image).float()

# Add batch dimension
tensor_image = tensor_image.unsqueeze(0)

# Move the input and model to the device
device = torch.device("cuda" if args_predict.gpu and torch.cuda.is_available() else "cpu")
tensor_image = tensor_image.to(device)
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Prediction
with torch.no_grad():
    output = model.forward(tensor_image)

# Calculate the probability
pb = torch.exp(output)
# Get the top k largest prob
top_prob, top_classes = pb.topk(args_predict.top_k, dim=1)

# Convert indices to class labels
idx_to_class = {val: key for key, val in checkpoint['class_to_idx'].items()}
top_labels = [cat_to_name[idx_to_class[cl.item()]] for cl in top_classes[0]]

# Display the prediction result
print(f"Top {args_predict.top_k} predictions:")
for label, prob in zip(top_labels, top_prob[0]):
    print(f"{label} with probability of {prob.item():.4f}")