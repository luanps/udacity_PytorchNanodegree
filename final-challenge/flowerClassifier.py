import torch
from torch import nn, optim, utils
from torchvision.transforms import transforms
from torchvision import datasets
#from PIL import Image
import pdb
import json

data_dir = 'flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

def load_image(data_dir, size=224):
    #image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize(size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((.485,.456,406),
                                            (.229,.224,.225))])

    data = datasets.ImageFolder(data_dir,transform=transform)
    dataloaders = utils.data.DataLoader(data, batch_size=4,shuffle=True)
    return dataloaders

train_data = load_image(train_dir,224)
val_data = load_image(valid_dir,224)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
pdb.set_trace()



