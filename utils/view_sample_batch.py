import torch.nn as nn
import timm
import torch
from torchsummary import summary 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt


input_dir = r"D:\IAAA_CMMD\manifest-1616439774456\test_dcm\dcom_2_png/"
meta_file = r"C:\Users\Omnia\Desktop\dcom_2_png/meta.csv"
train = pd.read_csv(meta_file)
train['classification'] = train['classification'].apply(lambda x: 0 if x == 'Benign' else 1)


# image preprocessing
def prepare_image(path, image_size = 256):
    
    # import
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # resize
    image = cv2.resize(image, (int(image_size), int(image_size)))
    #image = cv2.resize(image, (int(image_size), int(image_size)))


    # convert to tensor    
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image


# dataset
class EyeData(Dataset):
    
    # initialize
    def __init__(self, data, directory, transform = None):
        self.data      = data
        self.directory = directory
        self.transform = transform
        
    # length
    def __len__(self):
        return len(self.data)
    
    # get items    
    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.data.loc[idx, 'Img_name'] )
        image    = prepare_image(img_name)  
        image    = self.transform(image)
        label    = torch.tensor(self.data.loc[idx, 'classification'])
        return {'image': image, 'label': label}
    


##### EXAMINE SAMPLE BATCH
size = 256
# transformations
sample_trans = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                   transforms.ToPILImage(),
                                   transforms.ToTensor(),
                                  ])

# dataset
sample = EyeData(data       = train, 
                 directory  = input_dir,
                 transform  = sample_trans)

# data loader
sample_loader = torch.utils.data.DataLoader(dataset     = sample, 
                                            batch_size  = 8, 
                                            shuffle     = False, 
                                            num_workers = 0)

a =iter(sample_loader)
a1 = next(a)

# display images
for batch_i, data in enumerate(sample_loader):

    # extract data
    inputs = data['image']
    labels = data['label']#.view(-1, 1)
    print(inputs.shape)
    print(labels.shape)
    # create plot
    fig = plt.figure(figsize = (15, 7))
    for i in range(len(labels)):
        ax = fig.add_subplot(2, int(len(labels)/2), i + 1, xticks = [], yticks = [])
        #ax = fig.add_subplot(2,1,i ,xticks = [], yticks = [])     

        plt.imshow(inputs[i].numpy().transpose(1, 2, 0))
        ax.set_title(labels.numpy()[i])

    break



model =  timm.create_model('convnext_base', pretrained=True,num_classes=2)
