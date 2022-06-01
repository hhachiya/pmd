import torch
from torch import optim
import torch.nn.functional as F 
#from torch.autograd import Variable
from torchvision import transforms as T
from torchmetrics import JaccardIndex

import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser

from model.pmd import PMD
import dataset
import lovasz_losses as L
import pdb

#--------------
# functions

def dice_loss(pred, target, smooth = 1.):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def criterion(pred, target, target_edge, mode='DBCE'):
    if mode == 'DBCE':
       loss0 = F.binary_cross_entropy_with_logits(pred[0], target) + dice_loss(pred[0], target)
       loss1 = F.binary_cross_entropy_with_logits(pred[1], target) + dice_loss(pred[1], target)
       loss2 = F.binary_cross_entropy_with_logits(pred[2], target) + dice_loss(pred[2], target)
       loss3 = F.binary_cross_entropy_with_logits(pred[3], target) + dice_loss(pred[3], target)
       final_loss = F.binary_cross_entropy_with_logits(pred[5], target) + dice_loss(pred[5], target)
    elif mode == 'LH':
       loss0 = L.lovasz_hinge(pred[0], target)
       loss1 = L.lovasz_hinge(pred[1], target)
       loss2 = L.lovasz_hinge(pred[2], target)
       loss3 = L.lovasz_hinge(pred[3], target)
       final_loss = L.lovasz_hinge(pred[5], target)

    edge_loss = F.binary_cross_entropy_with_logits(pred[4], target_edge)

    loss = loss0 + loss1 + loss2 + loss3 + 5*edge_loss + 2*final_loss

    return loss

def save_model(model, optimizer, loss, path):
    torch.save({
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'loss': loss,
    }, f"{path}")

def load_model(model, optimizer, path):
    checkpoint = torch.load(f"{path}")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['loss']

    return best_loss

def parse_args():
    parser = ArgumentParser(description="pmd trainer")
    parser.add_argument('-data',type=str,default='split',help='name of dataset, e.g., split and theta, default: split')
    parser.add_argument('-is_load','--is_load',action='store_true')

    return parser.parse_args()


def comp_IoU(pred, target, thre=0.5):
    pred = (torch.sigmoid(pred) > thre).long()
    return JaccardIndex(num_classes=2)(pred, target.int())
#--------------

args = parse_args()

#--------------
# setting

img_size = 416
train_ratio = 0.8
batch_size = 15
epoch_num = 50 
patience_cnt = 5

# gpu
device_ids = [0]
torch.cuda.set_device(device_ids[0])

# file path
if args.data == 'split':
    data_dir = '../data/PMD_split/PMD'
    postfix = 'split'
    load_model_path = 'model/pmd_bak.pth'
elif args.data == 'theta':
    postfix = 'theta'
    data_dir = '../data/theta'
    load_model_path = 'model/pmd_split.pth'

save_model_path = f'model/pmd_{postfix}.pth'
result_dir = 'result'
x_train_dir = os.path.join(data_dir, 'train/image')
y_train_dir = os.path.join(data_dir, 'train/mask')
y_edge_train_dir = os.path.join(data_dir, 'train/edge')
#--------------


#--------------
# data

# transformation for image
img_transform = T.Compose([
    T.Resize((img_size, img_size)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# transformation for mask,mask_edge
img_label_transform = T.Compose({
    T.Resize((img_size,img_size)),
    T.ToTensor()
})

# creating data set
dataset = dataset.PmdDataset(x_train_dir,y_train_dir,y_edge_train_dir,img_transform,img_label_transform)
train_size = int(dataset.__len__() * train_ratio) 
val_size   = dataset.__len__() - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
#--------------

#--------------
# evaluation


train_losses = []
val_losses   = []
train_ious  = []
val_ious    = []
#--------------

#--------------
# model
model = PMD().cuda(device_ids[0])
optimizer = optim.Adam(model.parameters())

best_loss_cnt = 0
if args.is_load:
    best_loss = load_model(model, optimizer, load_model_path)
else:
    best_loss = float("inf")
#--------------

#--------------
# train
print("training start...")
model.train()
for epoch in range(epoch_num):
    print('epoch: {}'.format(epoch+1))

    torch.backends.cudnn.benchmark = True

    # evaluation
    train_loss = 0
    train_iou = 0
    cnt = 0

    # progress bar
    #bar = tqdm(total = train_size)
    #bar.set_description('Progress rate')    
    
    #for image, target, target_edge in tqdm(torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)):
    for image, target, target_edge in torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True):
        
        # initialize grad
        optimizer.zero_grad()
        
        # predict & update params
        pred = model(image.cuda()) 
        loss = criterion(pred, target.cuda(), target_edge.cuda())
        loss.backward()
        optimizer.step()
        
        # increment evaluation
        train_loss += loss.item()
        train_iou  += comp_IoU(pred[5].cpu(),target)
        cnt += 1

    # average evaluation
    train_loss /= cnt
    train_iou /= cnt
    print("train loss:{}, IoU:{}".format(train_loss,train_iou))
    train_losses.append(train_loss)
    train_ious.append(train_iou)
   
    # validation
    val_iou = 0
    val_loss = 0
    with torch.no_grad():
        #for image, target, target_edge in tqdm(torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=2, pin_memory=True)):                   
        for image, target, target_edge in torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=2, pin_memory=True):

            # prediction
            pred = model(image.cuda())

            # increment evaluation
            loss = criterion(pred, target.cuda(), target_edge.cuda())
            val_loss += loss.item()
            val_iou += comp_IoU(pred[5].cpu(), target) 

    # average evaluation
    val_iou /= len(val_dataset) 
    val_loss /= len(val_dataset)
    print("val Loss:{}, IoU:{}".format(val_loss,val_iou))
    val_losses.append(val_loss)
    val_ious.append(val_iou)

    # save & plot
    if epoch % 1 == 0:

        # save model
        if best_loss > val_loss:
            best_loss = val_loss
            best_loss_cnt = 0
            print(f"save model... loss:{val_loss}.")
            save_model(model, optimizer, val_loss, save_model_path)
        else:
            print(f"best_loss_cnt:{best_loss_cnt}.")
            best_loss_cnt += 1
        
        # save loss
        with open(f"{result_dir}/loss_iou_{postfix}.pkl",'wb') as w:
            pickle.dump(train_losses, w)
            pickle.dump(val_losses, w)
            pickle.dump(train_ious, w)
            pickle.dump(val_ious, w)

        # plot loss & iou
        fig = plt.figure()
        fig.add_subplot(2,1,1)
        plt.plot(train_losses, label='train loss')
        plt.plot(val_losses, label='val loss')
        plt.legend()

        fig.add_subplot(2,1,2)
        plt.plot(train_ious, label='train iou')
        plt.plot(val_ious, label='val iou')
        plt.legend()
        plt.xlabel('epoch')
        plt.savefig(f"{result_dir}/loss_iou_{postfix}.png")

        if best_loss_cnt > patience_cnt:
            break

save_model(model, optimizer, val_loss, save_model_path)
#--------------
