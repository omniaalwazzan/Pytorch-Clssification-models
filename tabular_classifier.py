#### Prams

EPOCHS = 100
BATCH_SIZE = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 2

import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from tqdm.notebook import tqdm
from ipywidgets import IntProgress
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report,f1_score,roc_curve, roc_auc_score
import sklearn.metrics as metrics

from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve,auc

from random import random
from random import seed

from init_weights import *
# seed the generator
#seed(7)

df  =  pd.read_csv(r"D:\IAAA_CMMD\manifest-1616439774456\CMMD_clinicaldata_revision_1.csv")
df.set_index('ID1', inplace=True)
bening_ = df.loc[df['classification'] == 'Benign', 'number'].sum()
Malignant_ = df.loc[df['classification'] == 'Malignant', 'number'].sum()

print(f'Nr. of Benign MRI (per-patient) in all dataset: {bening_}')
print(f'Nr. of Malignant MRI (per-patient) in all dataset: {Malignant_}')
df = df.drop(columns=['number','subtype'])
df.columns


#### Data imputations 

# age impute
df.iloc[:,1]= (df.iloc[:, 1]- df.iloc[:, 1].mean()) / df.iloc[:, 1].std()
# Encode the categorical variable abnormality and LeftRight
le = LabelEncoder()
df['abnormality'] = le.fit_transform(df['abnormality'])
df['LeftRight'] = le.fit_transform(df['LeftRight'])
# change the class label to 0 or 1
df['classification'] = df['classification'].apply(lambda x: 0 if x == 'Benign' else 1)

###### Data Split

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
y.head()
X
NUM_FEATURES = len(X.columns)

# Split into train+val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)

# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)


### Visualize 

def get_class_distribution(obj):
    count_dict = {
        "Benign": 0,
        "Malignant": 0,
    }
    
    for i in obj:
        if i == 0: 
            count_dict['Benign'] += 1
        elif i == 1: 
            count_dict['Malignant'] += 1   
        else:
            print("Check classes.")
            
    return count_dict

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,7))
# Train
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_train)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[0]).set_title('Class Distribution in Train Set')
# Validation
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_val)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[1]).set_title('Class Distribution in Val Set')
# Test
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_test)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[2]).set_title('Class Distribution in Test Set')

### Data Generator ## 

class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)
train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())



## Data loading
# train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
# #a = iter(train_loader)
# #a1 = next(a)    
# #a1[1]
                     
# val_loader = DataLoader(dataset=val_dataset, batch_size=1)
# test_loader = DataLoader(dataset=test_dataset, batch_size=1)


## MODEL ##
class Linear_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Linear_ = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            nn.ReLU(inplace=True),
            nn.LayerNorm(out_channels)
            )

    def forward(self, x):
        return self.Linear_(x)
    
class EHR(nn.Module):
    def __init__(self, h_dim=64):
        super(EHR, self).__init__()
        self.layer_1 = Linear_Layer(3, 20) # with subtype in =4, without in=3
        self.layer_2 = Linear_Layer(20, 10)
        self.layer_3 = Linear_Layer(10, 2)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.dropout(x) 
        x = self.layer_3(x)
        x = self.dropout(x)
        return x
    
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
model = EHR()
model.to(device)
print("Number of Trainable Parameters: %d" % count_parameters(model))
init_type = ['normal' , 'xavier' , 'kaiming' , 'orthogonal' , 'max']
init_net(model, init_type[2])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#### Sampling Wieghts
target_list = []
for _, t in train_dataset:
    target_list.append(t)
    
target_list = torch.tensor(target_list)


class_count = [i for i in get_class_distribution(y_train).values()]
class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
print('class weights for EHR data',class_weights)

class_weights_all = class_weights[target_list]

weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          sampler=weighted_sampler)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

print("Begin training.")

for e in tqdm(range(1, EPOCHS+1)):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = model(X_train_batch)
        
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
        
        
    # VALIDATION    
    with torch.no_grad():
        
        val_epoch_loss = 0
        val_epoch_acc = 0
        
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
                              
    
    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')
    
    
# Create dataframes
train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# Plot the dataframes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')

### Testing

y_pred_list = []
y_test_pred_arr = []
#props= None
y_prob = []
with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        #probs = torch.nn.functional.softmax(y_test_pred, dim=1)
        #y_prob=y_prob.np.append(probs.detach().cpu().numpy())
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
        #y_prob = np.concatenate(y_prob)

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

# confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list))

# sns.heatmap(confusion_matrix_df, annot=True)

cm = confusion_matrix(y_test, y_pred_list)
classes= ['Benign', 'Malignant']
#classes= ['0']
tick_marks = np.arange(len(classes))
df_cm = pd.DataFrame(cm, index = classes, columns = classes)
plt.figure(figsize 
           = (7,7))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
plt.xlabel("Predicted label", fontsize = 20)
plt.ylabel("Ground Truth", fontsize = 20)

#print(classification_report(y_test, y_pred_list))

# from collections import Counter
# Counter(y_test) # y_true must be your labels

# Compute ROC curve and ROC area for each class


test_y = y_test
y_pred = y_pred_list

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_list, pos_label=2)
roc_auc = metrics.auc(fpr, tpr)


# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()


print(classification_report(y_test, y_pred_list))
print ('F1-score micro equals: ', f1_score(y_test, y_pred_list, average='micro'))
print ('F1-score macro equals: ', f1_score(y_test, y_pred_list, average='macro'))
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_pred_list)
metrics.auc(false_positive_rate1, true_positive_rate1)
print('roc_auc_score: ', roc_auc_score(y_test, y_pred_list))
print('AUC: ', metrics.auc(false_positive_rate1, true_positive_rate1))

fig4 =plt.figure()
fig4.add_subplot(111)
plt.title('Receiver Operating Characteristi')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


precision, recall, thresholds = precision_recall_curve(y_test, y_pred_list)
print ("precision-recall AUC is: ",metrics.auc(recall, precision))

# calculate precision-recall AUC
lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_pred_list)
lr_f1 = f1_score(y_test, y_pred_list)
# summarize scores
print('F1-score: f1=%.3f ' % (lr_f1))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
# disp.plot()
