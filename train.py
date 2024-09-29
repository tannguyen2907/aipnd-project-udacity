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

def get_device(device):
    return torch.device("cuda:0" if device else "cpu")

def build_pre_trained_model(device, arch, hidden_units):
    if arch == 'vgg16':
        pre_trained_model = models.vgg16(pretrained=True)
        init_value = 25088
    else:
        pre_trained_model = models.alexnet(pretrained=True)
        init_value = 9216

    pre_trained_model.features.require_grad = False

    pre_trained_model.classifier = nn.Sequential(nn.Linear(init_value,4096),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.5),
                                                 nn.Linear(4096,hidden_units),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.5),
                                                 nn.Linear(hidden_units,102),
                                                 nn.LogSoftmax(dim=1))
    pre_trained_model = pre_trained_model.to(device)

    return pre_trained_model

def do_train(pre_trained_model, training_loader, criterion , optimizer, device):
    total_step = len(training_loader)
    total_loss = 0
    print("Total training step: {}".format(total_step))
    for i, (inputs,labels) in enumerate(training_loader):
        print("Running step: {}".format(i+1))
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = pre_trained_model.forward(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        loss_item = loss.item()

        print("Current training loss: {:.4f}".format(loss_item))
        total_loss += loss_item
        print("Total training loss: {:.4f}".format(total_loss))

    return total_loss

def do_test(pre_trained_model, testing_loader, criterion, device):
    pre_trained_model.eval()
    correct = 0
    total = 0
    loss_total = 0
    count = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    print("Start verifying...")
    with torch.no_grad():
        for data in testing_loader:
            print("Verify step: {}".format(count+1))
            count = count + 1
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = pre_trained_model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return loss_total, (100*correct/total)

def save_check_point(model, optimizer, datasets, count, filename):
    torch.save({
        'count': count,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx' : datasets.class_to_idx,
        'classifier': model.classifier
    }, filename)

    if os.path.isfile(filename):
        print("Done saving checkpoint, filename: {}".format(filename))
    else:
        print("Error saving checkpoint.")

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for image classifier")
    parser.add_argument('data_dir', type=str, help='Data directory containing training, validation, and test sets')
    parser.add_argument('--save_dir', default='', type=str, help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'alexnet'], help='Model architecture, only accept 2 value: vgg16 or alexnet')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=1024, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--GPU', action='store_true', help='Use GPU for training if available')
    return parser.parse_args()

def main():
    args = parse_args()

    count = args.epochs
    device = get_device(args.GPU)
    print("Device using: {}".format(device))
    arch = args.arch
    hidden_units = args.hidden_units
    learning_rate = args.learning_rate

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    if not os.path.isdir(train_dir):
        print("Missing {}, exiting...".format(train_dir))
        return
    if not os.path.isdir(valid_dir):
        print("Missing {}, exiting...".format(valid_dir))
        return
    if not os.path.isdir(test_dir):
        print("Missing {}, exiting...".format(test_dir))
        return

    data_training_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])

    data_testing_validation_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])

    image_training_datasets = torchvision.datasets.ImageFolder(root = train_dir, transform = data_training_transforms)
    image_validation_datasets = torchvision.datasets.ImageFolder(root = valid_dir, transform = data_testing_validation_transforms)
    image_testing_datasets = torchvision.datasets.ImageFolder(root = test_dir, transform = data_testing_validation_transforms)

    training_dataloader = DataLoader(image_training_datasets, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(image_validation_datasets, batch_size=64)
    testing_dataloader = DataLoader(image_testing_datasets, batch_size=64)

    check_point_file_name = "{}/pre_trained_model.pth".format(args.save_dir)
    if args.save_dir == '':
        check_point_file_name = "pre_trained_model.pth"

    if os.path.isfile(check_point_file_name):
        print("Training model is existed at {}, please remove the file and re-run again, exiting...".format(check_point_file_name))
        return
    else:
        print("Not found training model, start building...")

    pre_trained_model = build_pre_trained_model(device, arch, hidden_units)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(pre_trained_model.classifier.parameters(), lr=learning_rate)

    for i in range(count):
        print("Start run: {} of {}".format(i+1, count))
        train_loss = do_train(pre_trained_model, training_dataloader, criterion , optimizer, device)
        val_loss, val_acc = do_test(pre_trained_model, validation_dataloader, criterion, device)
        print("Training Loss: {:.4f}".format(train_loss))
        print("Validation Loss: {:.4f} Validation Accuracy: {:.2f}%".format(val_loss, val_acc))
        if i == (count - 1):
            save_check_point(pre_trained_model, optimizer, image_training_datasets, count, check_point_file_name)
            break

if __name__ == "__main__":
    main()
