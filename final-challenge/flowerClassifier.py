import torch
from torch import nn, optim, utils
from torchvision import datasets, models, transforms
#from PIL import Image
import pdb
import json
import numpy as np

data_dir = 'flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# check if GPU is available 
train_on_gpu = torch.cuda.is_available() 
if(train_on_gpu): 
    print('Training on GPU!') 
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.') 



def load_image(data_dir, size=224):
    #image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize(size=256),
                                    transforms.CenterCrop(size=size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])])

    data = datasets.ImageFolder(data_dir,transform=transform)
    dataloaders = utils.data.DataLoader(data, batch_size=32,shuffle=True)
    return dataloaders

train_loader = load_image(train_dir,224)
valid_loader = load_image(valid_dir,224)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model = models.vgg16(pretrained=True)#.features
for param in model.parameters():
    param.requires_grad_ = False

model.classifier[-1] = nn.Sequential(
                     nn.Linear(4096,2048),
                     nn.ReLU(),
                     nn.Dropout(.5),
                     nn.Linear(2048,len(cat_to_name)))
#model.classifier[6] = nn.Linear(in_features=4096,out_features=len(cat_to_name))

print(model)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters())
optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
epochs = 50
valid_loss_min = np.Inf

if train_on_gpu:
    model = model.cuda()

for epoch in range(epochs):
    train_loss = 0.0
    valid_loss = 0.0


    if (train_on_gpu):
        model.cuda()
    model.train()

    for data, target in train_loader:
        if (train_on_gpu):
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)

    #validate model
    model.eval()
    for data, target in valid_loader:
        if (train_on_gpu):
            data, target = data.cuda(), target.cuda()
        out = model(data)
        loss = criterion(out, target)
        valid_loss += loss.item()*data.size(0)

    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    print('Epoch {} \tTraining Loss: {:.6f} \tValid Loss: {:.6f}'.format(epoch+1, train_loss, valid_loss))
    
    #save model if validation has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'model.pth')
        valid_loss_min = valid_loss

