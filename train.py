import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse

def main():
    # Command-line arguments for user input
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Dataset directory')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture: vgg16 or resnet18')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()

    # Data transformations and loading
    data_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])
    train_data = datasets.ImageFolder(args.data_dir + '/train', transform=data_transforms)
    valid_data = datasets.ImageFolder(args.data_dir + '/valid', transform=data_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    # Choose model architecture
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_features = 25088
    elif args.arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_features = 512

    # Freeze pre-trained model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define new classifier
    classifier = nn.Sequential(
        nn.Linear(input_features, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(args.hidden_units, 102),  # Adjust according to number of classes in your dataset
        nn.LogSoftmax(dim=1)
    )
    
    # Replace classifier
    if args.arch == 'vgg16':
        model.classifier = classifier
    elif args.arch == 'resnet18':
        model.fc = classifier

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters() if args.arch == 'vgg16' else model.fc.parameters(), lr=args.learning_rate)

    # Training on GPU
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(args.epochs):
        train_loss, valid_loss, accuracy = 0, 0, 0
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                accuracy += torch.sum(preds == labels).item()

        print(f"Epoch {epoch+1}/{args.epochs} - Training Loss: {train_loss/len(trainloader):.4f} - Validation Loss: {valid_loss/len(validloader):.4f} - Accuracy: {accuracy/len(validloader.dataset):.4f}")

    # Save the model checkpoint
    torch.save({'model_state_dict': model.state_dict(), 'classifier': model.classifier}, 'checkpoint.pth')

if __name__ == '__main__':
    main()

