from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import helper

#Network architecture
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        #flatten input tensor
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

#Normalize data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((.5,.5,.5),(.5,.5,.5))])

#Download and load training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',download=True,train=True,transform=transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=64,suffle=True)

#Download and load test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',download=True,train=False,transform=transform)
testLoader = torch.utils.data.DataLoader(testset, batch_size=64,suffle=True)

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for imgs, labels, in trainLoader:
        logps = model(imgs)
        loss = criterion(logps, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss}")


dataiter = iter(testloader)
imgs, labels = dataiter.next()
img = images[1]
ps = torch.exp(model(img))

helper.view_classify(img, ps, version='Fashion')

