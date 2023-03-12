

# instantiate model
import timm
from torch.nn import functional as F
import torch.nn as nn
import numpy as np 
import torch.optim as optim
import torch

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = timm.create_model('convnext_base', num_classes = 2)
#timm.create_model('convnext_base', pretrained=True,num_classes=2)

# This code snippet inspired by: https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2022/03/29/discriminative-lr.html

# save layer names
layer_names = []
for idx, (name, param) in enumerate(model.named_parameters()):
    layer_names.append(name)
    print(f'{idx}: {name}')
    
layer_names.reverse()
layer_names[0:5]


# learning rate
lr      = 1e-2
lr_mult = 0.9

# placeholder
parameters      = []
prev_group_name = layer_names[0].split('.')[0]

# store params & learning rates
for idx, name in enumerate(layer_names):
    
    # parameter group name
    cur_group_name = name.split('.')[0]
    
    # update learning rate
    if cur_group_name != prev_group_name:
        lr *= lr_mult
    prev_group_name = cur_group_name
    
    # display info
    print(f'{idx}: lr = {lr:.6f}, {name}')
    
    # append layer parameters
    parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                    'lr':     lr}]
    


# #model = timm.l('convnext_base', num_classes = 2)

# all_densenet_models = timm.list_models('*convnext*')
# model =  timm.create_model('convnext_small_384_in22ft1k', pretrained=True,num_classes=2)


optimizer = optim.Adam(model.parameters())
loss_fn1 = nn.CrossEntropyLoss()

        
        
        


valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_loader)
for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_,img_n) in enumerate(train_loader):
        #data_, target_ = data_.permute(0,3,1, 2).to(device), target_.to(device)
        data_, target_ = data_.to(device), target_.to(device)
        
        with torch.set_grad_enabled(True):


            optimizer.zero_grad()
    
            outputs = model(data_.float())
            loss = loss_fn1(outputs, target_)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            _,pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==target_).item()
            total += target_.size(0)
            if (batch_idx) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        model.eval()
        for data_t, target_t, img_n in (val_loader):
            #data_t, target_t = data_t.permute(0,3,1, 2).to(device), target_t.to(device)
            data_t, target_t = data_t.to(device), target_t.to(device)

            outputs_t = model(data_t.float())
            loss_t = loss_fn1(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t/total_t)
        val_loss.append(batch_loss/len(val_loader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')


        if network_learned:
            valid_loss_min = batch_loss
            torch.save(model.state_dict(), 'mhaF.pt')
            print('Improvement-Detected, save-model')
    model.train()
