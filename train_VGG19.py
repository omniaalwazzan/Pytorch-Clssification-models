images_folder = '/data/DERI-MMH/datasets/ROIs' # in case of tra test on rois
test_images_folder ='/data/DERI-MMH/datasets/all_patches' # k= 3 with act ELU at the end not good

path_to_save_check_points='/data/DERI-MMH/datasets/early_stopping/cnn_classifier' + '/cnn_f4' # wi means wieght init was used
path_to_checkpoints=path_to_save_check_points

# FOLD 4




batch_size = 4
Max_Epochs = 10
LEARNING_RATE = 5e-5 # 0.005
WEIGHT_DECAY = 5e-6 # 0.0005
patience = 7
perplexity = 60
n_iter =300
NUM_WORKERS=0
PIN_MEMORY=True


from sklearn import manifold
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import PIL 
import cv2
import numpy as np
from PIL import Image 
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torchvision import transforms
from torch import nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchinfo import summary
import random
import torch.nn.functional as F
from tqdm import tqdm
#from mhaSelf_moaba import *
fold = os.path.basename(os.path.normpath(train_clinical_path))[:-4]
#print(fold)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


train_transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.Resize((512,512)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(self, df1, images_folder, transform=None):
        self.df1 = df1
        self.images_folder = images_folder
        self.images_name = df1['Long Slide ID']       
        self.y = df1['Grade']
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        img_path=os.path.join(self.images_folder,
                              self.images_name[index])
        image=Image.open(img_path)
        
        y_label = self.y[index]
  
        if self.transform is not None:
            image = self.transform(image)
        return (image,y_label,self.images_name[index])

def load_data(df1, image_dir):
    train_data =CustomDataset(df1, image_dir,transform=train_transforms)
    
    return train_data

def Data_Loader(df1,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = CustomDataset(df1=df1 ,images_folder=images_folder,transform=train_transforms)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader



df_clinical_train = pd.read_csv(train_clinical_path)
#val_loader = pd.read_csv(val_clinical_path)
df_clinical_test1 = pd.read_csv(test_clinical_path)



train_loader = Data_Loader(df_clinical_train,images_folder,batch_size)
#Fusion_val_loader = Data_Loader(df_clinical_val,images_folder,9)
val_loader = Data_Loader(df_clinical_test1,test_images_folder,batch_size)

print(f"length of traing loader is:{ len(train_loader)}") #this shoud be = Total_images/ batch size
print(f"lenght of validation loader {len(val_loader)}")


### Specify all the Losses (Train+ Validation), and Validation Dice score to plot on learing-curve
avg_train_losses = []   # losses of all training epochs
avg_valid_losses = []  #losses of all training epochs

###  1- To stop the training before model overfits 
class EarlyStopping:
    def __init__(self, patience=None, verbose=True, delta=0,  trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score1 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.max_score = 0
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss,val_metric):

        score = -val_loss
        score1=-val_metric

        if (self.best_score is None) and (self.best_score1 is None):
            self.best_score = score
            self.best_score1 = score1
            self.verbose_(val_loss,val_metric)
        elif (score < self.best_score + self.delta) or (score1 > self.best_score1 + self.delta):
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score1 = score1
            self.verbose_(val_loss,val_metric)
            self.counter = 0

    def verbose_(self, val_loss,val_metric):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.trace_func(f'Validation metric increased ({self.max_score:.6f} --> {val_metric:.6f}).')
        self.val_loss_min = val_loss
        self.max_score = val_metric
    
    

model_urls = {
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
}



class PathNet(nn.Module):

    def __init__(self, features, path_dim=64, act=None, num_classes=3):
        super(PathNet, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, path_dim),
            nn.ReLU(True),
            nn.Dropout(0.05)
        )

        self.linear = nn.Linear(path_dim, num_classes)
        self.act = act

        #self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        #self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        #dfs_freeze(self.features)

    def forward(self,x):
        #x = kwargs['x_path']
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        features = self.classifier(x)
        hazard = self.linear(features)

        if self.act is not None:
            hazard = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return  features,hazard
        #return  hazard



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {

    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



def get_vgg(arch='vgg19_bn', cfg='E', act=None, batch_norm=True, label_dim=3, pretrained=True, progress=True):
    model = PathNet(make_layers(cfgs[cfg], batch_norm=batch_norm), act=act, num_classes=label_dim)
    
    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        for key in list(pretrained_dict.keys()):
            if 'classifier' in key: pretrained_dict.pop(key)

        model.load_state_dict(pretrained_dict, strict=False)
        print("Initializing Path Weights")

    return model



model = get_vgg().to(device=DEVICE,dtype=torch.float32)

def train_fn(loader_train,loader_valid, model, optimizer,loss_fn1, scaler):
    train_losses = [] # loss of each batch
    valid_losses = []  # loss of each batch
    loop = tqdm(loader_train)
    model.train()
    for batch_idx, (img1,gt1,label) in enumerate(loop):
        img1 = img1.to(device=DEVICE,dtype=torch.float32)  
        gt1 = gt1.to(device=DEVICE,dtype=torch.long)
       
        # forward
        with torch.cuda.amp.autocast():
            feat ,out1= model(img1)   
            loss = loss_fn1(out1, gt1)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        train_losses.append(loss.item())
    
    loop_v = tqdm(loader_valid)
    model.eval()
    for batch_idx, (img1,gt1,label) in enumerate(loop_v):
        img1 = img1.to(device=DEVICE,dtype=torch.float32)
        gt1 = gt1.to(device=DEVICE,dtype=torch.long)
        # forward
        with torch.no_grad():
            feat ,out1 = model(img1)   
            loss = loss_fn1(out1, gt1)
        loop_v.set_postfix(loss=loss.item())
        valid_losses.append(loss.item())    
    train_loss_per_epoch = np.average(train_losses)
    valid_loss_per_epoch = np.average(valid_losses)
    ## all epochs
    avg_train_losses.append(train_loss_per_epoch)
    avg_valid_losses.append(valid_loss_per_epoch)
    
    return train_loss_per_epoch,valid_loss_per_epoch

def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
       
epoch_len = len(str(Max_Epochs))
early_stopping = EarlyStopping(patience=10, verbose=True)


def main():
    #model = model.to(DEVICE=DEVICE,dtype=torch.float)
    loss_fn1 = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(Max_Epochs):
        if epoch<10:
            LEARNING_RATE = 0.00005
        if epoch>10:
            LEARNING_RATE = 0.000005
          
        optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
        train_loss,valid_loss=train_fn(train_loader,val_loader, model, optimizer, loss_fn1,scaler)
        
    
        
        print_msg = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)

        #dice_score = check_Dice_Score(val_loader, model, DEVICE=DEVICE)
        
        
        #avg_valid_DS.append(dice_score.detach().cpu().numpy())
        
        early_stopping(valid_loss, train_loss)
        if early_stopping.early_stop:
            print("Early stopping Reached at  :",epoch)
            
            ### save model    ######
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

            break
            
if __name__ == "__main__":
    main()
def eval_plt(test_loader,model,device):
    
    truelabels = []
    predictions = []
    proba = []
    pre = []
    
    with torch.no_grad():
        model.eval()
        for data, target,img_name in test_loader:
            data = data.to(device=device,dtype=torch.float)
            target = target.to(device=device,dtype=torch.float)
            truelabels.extend(target.cpu().numpy())
    
            feat,output = model(data)
            probs = F.softmax(output, dim=1)[:, 1]# assuming logits has the shape [batch_size, nb_classes]
            #top_p, top_class = prob.topk(1, dim = 1)
            preds = torch.argmax(output, dim=1) # this to plot the confusion matrix
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted.cpu().numpy())
            proba.extend(probs.cpu().numpy())
            pre.extend(preds.cpu().numpy())
            from sklearn.metrics import f1_score
        print('F1-score micro for MLP classifer:')
        print(f1_score(truelabels, predictions, average='micro'))
        print(classification_report(truelabels, predictions))
            
        cm = confusion_matrix(truelabels, pre)
        classes= ['GradeII', 'GradeIII', 'GradeIV']
        tick_marks = np.arange(len(classes))
        
        df_cm = pd.DataFrame(cm, index = classes, columns = classes)
        plt.figure(figsize = (7,7))
        sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
        plt.xlabel("Predicted label", fontsize = 20)
        plt.ylabel("Ground Truth", fontsize = 20)
        plt.show()
        return truelabels,predictions, proba, pre
    
truelabels,predictions, proba, pre= eval_plt(val_loader,model,DEVICE)

def get_representations(model, iterator,device):
    
    model.eval().to(device)

    outputs = []
    labels = []
    feature = []

    with torch.no_grad():

        for (x, y,imge_name) in tqdm(iterator):

            x = x.to(device)
            y = y.to(device)
            
            y = y.data.cpu()

            feat ,y_pred = model(x)

            outputs.append(y_pred.data.cpu())
            feature.append(feat.data.cpu())

            labels.append(y)

    outputs = torch.cat(outputs, dim=0)
    feature = torch.cat(feature, dim=0)
    labels = torch.cat(labels, dim=0)

    return outputs,feature ,labels

def get_tsne(data, n_components=2, n_images=None):

    if n_images is not None:
        data = data[:n_images]

    tsne = manifold.TSNE(n_components=n_components, random_state=0, perplexity=perplexity, n_iter=n_iter)
    tsne_data = tsne.fit_transform(data)
    return tsne_data

outputs, feature,labels = get_representations(model, val_loader,DEVICE)

def plot_representations(data, labels, classes, n_images=None):

    if n_images is not None:
        data = data[:n_images]
        labels = labels[:n_images]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10')
    handles, labels = scatter.legend_elements()
    ax.legend(handles=handles, labels=classes)
    plt.savefig('tsne_'+fold+'.png')


N_IMAGES = 2_000
classes = [0,1,2]
output_tsne_data = get_tsne(feature, n_images=N_IMAGES)
plot_representations(output_tsne_data, labels, classes, n_images=N_IMAGES)
