import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

# Path where model should be saved locally
PATH = './cifar_net.pth'

class ImageClassifier:
    def __init__(self):
        print(f'init image classifier')

    # Load and normalize CIFAR10
    '''def load_and_normalize_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        batch_size = 4

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

        #classes = ('plane', 'car', 'bird', 'cat',
        #        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')'''

    # Train the network
    def train(self, num_epochs):

        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {device}")

        # Load and normalize CIFAR10 data
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

        # Define a Convolutional Neural Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        net = Net().to(device)

        # Define a Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # load train and test data
        batch_size = 4
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

        # start training
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')
        #torch.save(net.state_dict(), PATH)

        # Test the network on the test data
        #net = Net()
        #net.load_state_dict(torch.load(PATH, weights_only=True))
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


if __name__ == "__main__":
    image_classifier = ImageClassifier()
    image_classifier.train(num_epochs=2)
