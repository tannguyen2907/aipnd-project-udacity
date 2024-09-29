import torchvision
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json

def get_device(device):
    return torch.device("cuda:0" if device else "cpu")

def load_check_point_model(filename):
    loaded_model = torch.load(filename)
    if loaded_model['arch'] == "vgg16":
        check_point_model = models.vgg16(pretrained=True)
    elif loaded_model['arch'] == "alexnet":
        check_point_model = models.alexnet(pretrained=True)
    else:
        raise Exception("Model arch not support, only vgg16 or alexnet is suported, exiting...")
    check_point_model.class_to_idx = loaded_model['class_to_idx']
    check_point_model.classifier = loaded_model['classifier']
    check_point_model.load_state_dict(loaded_model['model_state_dict'])

    return check_point_model

def process_image(image):
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img.resize((256, 256))

    width = img.width
    height = img.height

    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    crop_img = img.crop((left_margin, bottom_margin, left_margin +224, bottom_margin + 224))

    np_image = np.array(crop_img) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    standard = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / standard

    np_image = np_image.transpose((2, 0, 1))

    return np_image

def predict(image_path, model, topk, device):

    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0)
    image = image.to(device)

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        probabilities = torch.exp(model(image))
        top_probabilities, top_classes = probabilities.topk(topk, dim=1)

    index_to_class = {value:key for key, value in model.class_to_idx.items()}
    top_classes = [index_to_class[i] for i in top_classes.tolist()[0]]

    return top_probabilities[0].cpu().numpy(), top_classes

def parse_args():
    parser = argparse.ArgumentParser(description="Predict the class of an image")
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint_path', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K predictions')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    return parser.parse_args()

def main():
    args = parse_args()

    image_path = args.image_path
    checkpoint_path = args.checkpoint_path
    top_k = args.top_k
    device = get_device(args.gpu)
    print("Device using: {}".format(device))
    category_names = args.category_names
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_check_point_model(checkpoint_path)
    top_probabilities, top_classes = predict(image_path, model, top_k, device)
    class_names = [cat_to_name[cls] for cls in top_classes]
    print("Flower name predict: {}".format(class_names))
    print("Top K: {}".format(top_classes))

if __name__ == "__main__":
    main()
