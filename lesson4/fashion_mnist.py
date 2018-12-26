from torch import nn, optim
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

#Network architecture
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        #20% probability Dropout 
        self.dropout = nn.Dropout(p=.2)

    def forward(self, x):
        #flatten input tensor
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

#Normalize data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((.5,.5,.5),(.5,.5,.5))])

#Download and load training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',download=True,train=True,transform=transform)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=64,shuffle=True)

#Download and load test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',download=True,train=False,transform=transform)
testLoader = torch.utils.data.DataLoader(testset, batch_size=64,shuffle=True)

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30
steps = 0
train_losses, test_losses = [], []
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
        
        test_loss = 0
        acc = 0
        #Disable gradients for validation
        with torch.no_grad():
            #set model to evaluation mode (w/o dropout)
            model.eval()

            for imgs, labels in testLoader:
                log_ps = model(imgs)
                test_loss += criterion(log_ps, labels)
                #class probability
                ps = torch.exp(log_ps)
                #most likely classes
                top_p, top_class = ps.topk(1, dim=1)
                #check if top classes matches with the labels
                equals = top_class == labels.view(*top_class.shape)
                #convert 'equals' to a float tensor
                acc += torch.mean(equals.type(torch.FloatTensor))

        train_losses.append(running_loss/len(trainLoader))
        test_losses.append(test_loss/len(testLoader))
        print("Epoch: {}/{}.. ".format(e+1,epochs),
                "Training Loss: {:.3f}.. ".format(running_loss/len(trainLoader)),
                "Test Loss: {:.3f}.. ".format(test_loss/len(testLoader)),
                "Test Acc: {:.3f}.. ".format(acc/len(testLoader)))

        #set model back to train mode
        model.train()


#Plot Train and Validation loss
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

#Save model to file
torch.save(model.state_dict(), 'checkpoint.pth')
print("Model:' \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())



