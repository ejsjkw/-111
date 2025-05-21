import cv2
import os
import numpy as np
import torch as tf
import torch.nn as nn
import torchvision as tfv
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image as  PILImage 
from PyQt5.QtWidgets import * #QWidget, QSizePolicy, QMainWindow, QApplication
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt

from handnet_params_funcs import *

DIGIT_CLASSNUMS = 18
DIGIT_SIZEWH = 38
DIGIT_TRAIN_BATCHSIZE = 10
DIGIT_DFT_SCALE = 0.75
DIGIT_IMG_ROOT = "./imgs/"



def digit_DescToImage(desc,targetWH=DIGIT_SIZEWH,scale=DIGIT_DFT_SCALE): #targetWH 28*28
#digit_descriptor ={} # {"lb":0,"color":(255,244,30),"box":(10,10,100,100),"seqs":[[ [1,1],[2,2] ],  [ [4,2],[10,3],[2,5] ]] } #pts:笔画数*笔画长度*2（点对）
    img =  np.zeros((targetWH, targetWH,3),dtype="uint8")
    box = desc["box"]
    scaleR = targetWH/max(box[2],box[3])
    scaleR = scaleR*scale
    tcx,tcy = targetWH/2,targetWH/2
    scx,scy = box[0]+(box[2]/2), box[1]+(box[3]/2)
    
    seqs = desc["seqs"]
    
    t_seqlist = []
    for seq in seqs:
        t_seq = []
        for pt in seq: #(12,12)
            dx,dy = pt[0]-scx,pt[1]-scy
            dx,dy = dx*scaleR+tcx,dy*scaleR+tcy
            t_seq.append((round(dx),round(dy)))         
        t_seqlist.append(t_seq)
    #print("seqs:",seqs)
    #print("t_seqlist:",t_seqlist)
    draw_immain_seqlist(img, t_seqlist,(255,255,255),1)
    return img





def digit_predictFromDesc(trainer,desc):
    global DIGIT_SIZEWH,ui_idx_to_label,ui_label_to_idx
    if desc["lb"]>=0: 
        print("Digit is labelled!")
        return
    lenet5_trans=transforms.Compose([
        transforms.Grayscale(),#此变换将以给定的概率水平（随机）翻转图像。通过参数“p”来设置这个概率。p的默认值为0.5。
        transforms.Resize([DIGIT_SIZEWH, DIGIT_SIZEWH]),
        transforms.ToTensor()
    ])

    pilim = PILImage.fromarray(np.array(desc["im"]))   #这里ndarray_image为原来的numpy数组类型的输入

    im = lenet5_trans(pilim).unsqueeze(0)  #[1,1,38,38]
    lb = trainer.netpredict(im,tf.device("cpu"))
    lb = lb.to("cpu").item()
    lb = trainer.net_idx_to_label[lb]
    lb = ui_label_to_idx[lb]
    
    desc["lb"] = lb
    lb = ui_idx_to_label[lb]
    #print("Dignet predict:",lb) #debug
    return 


def check_digits_mouseclk(pt,digstruct):
    for desc in digstruct:
        if point_in_rect(desc["box"],pt)>0: 
            desc["sel"] = True
        else : desc["sel"] = False
    for desc in digstruct:
        if desc["sel"]==True: return desc
    return None

def unlabel_selected_digit(pt, digstruct):
    for desc in digstruct:
        if point_in_rect(desc["box"],pt)>0: 
            desc["sel"] = True
        else : desc["sel"] = False
    for desc in digstruct:
        if desc["sel"]==True: desc["lb"] = -1
    return


# ui_idx_to_label={0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",
#                             10:"add",11:"sub",12:"mul",13:"div",14:"dot",15:"leftk",16:"eq",17:"rightk"}
#DIGIT_IMG_ROOT = "./imgs/"

def get_imagefiles_count():
    global DIGIT_IMG_ROOT
    global ui_idx_to_label

    im_count = [ 0 for i in range(18)]
    for i in range(18):
        path = os.path.join(DIGIT_IMG_ROOT,ui_idx_to_label[i])
        files = os.listdir(path)
        for f in files:  
            n,png = f.split('.')   
            if png != "png": continue
            n = int(n) 
            if n>im_count[i]: im_count[i] = n  
    return im_count

def show_imagefiles_num(ui_lb,im_count):
    global ui_idx_to_label

    lbstr = ""
    for i,c in zip(im_count, ui_idx_to_label.values()):
        lbstr = lbstr+c+":"+str(i)+"|"
    ui_lb.setText(lbstr)
    return
            
def save_new_digits(digstruct):
    global DIGIT_IMG_ROOT
    global ui_idx_to_label
    if len(digstruct)<=0:
        print("No files to write!")
        return 
    
    im_count =  get_imagefiles_count()

    for desc in digstruct:
        lb = desc["lb"]
        if lb <0: continue
        path = os.path.join(DIGIT_IMG_ROOT, ui_idx_to_label[lb])
        name = str(im_count[lb]+1)+".png"
        im_count[lb] = im_count[lb]+1
        pathname = os.path.join(path,name)

        cv2.imwrite(pathname,desc["im"],[cv2.IMWRITE_PNG_COMPRESSION,0])
        print("write file:",pathname)
    
    return im_count
    # for desc in digstruct:
    #     lb = desc["lb"]
    #     if lb<0: continue
        
def show_labeledinfo(ui_lb, digstruct):
    cnt = 0
    for desc in digstruct:
        lb = desc["lb"]
        if lb <0: continue
        cnt +=1
    ui_lb.setText(f"Labeled new images:{cnt}")
    return

class class_LeNet(nn.Module):
    def __init__(self):
        super(class_LeNet,self).__init__()
        self.cv1 = nn.Conv2d(1,6,3)
        self.p1 = nn.MaxPool2d(2,2)
        self.cv2 = nn.Conv2d(6,16,3)
        self.p2 = nn.MaxPool2d(2,2)
        self.fc3 = nn.Linear(16*8*8, 140)
        self.fc4 = nn.Linear(140,90)
        self.fc5 = nn.Linear(90,18)
    def forward(self,x):
        x = self.p1(tf.relu(self.cv1(x)))
        x = self.p2(tf.relu(self.cv2(x)))
        x = x.view(x.size(0), -1)
        x = tf.relu(self.fc3(x))
        x = tf.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    
class class_digit_trainer():
    def __init__(self):  #"cuda:0"  or "cpu"
        
        #########
        self.modelnet = class_LeNet()
        self.lossfunc = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.modelnet.parameters(), lr=0.01, momentum=0.9)
        self.lenet5_trainloader = None
        self.net_idx_to_label={}
        self.net_label_to_idx = {}
        self.loadnet()
        self.loaddata()
        return
    def netpredict(self,x,device=tf.device("cuda:0")): #x:(N, 1,38,38)
        x = x.to(device)
        self.modelnet.to(device)
        out = self.modelnet(x)
        #print("outpredict shape",out.shape)
        pred = tf.softmax(out,dim=1)
        return tf.argmax(pred,dim=1)
    
    def loaddata(self):
        global DIGIT_TRAIN_BATCHSIZE
        lenet5_trans=transforms.Compose([
            transforms.Grayscale(),#此变换将以给定的概率水平（随机）翻转图像。通过参数“p”来设置这个概率。p的默认值为0.5。
            transforms.Resize([DIGIT_SIZEWH, DIGIT_SIZEWH]),
            transforms.ToTensor()
        ])
        lenet5_imagefolder = tfv.datasets.ImageFolder(DIGIT_IMG_ROOT, transform=lenet5_trans)
        self.lenet5_trainloader = tf.utils.data.DataLoader(lenet5_imagefolder,batch_size=DIGIT_TRAIN_BATCHSIZE,shuffle=True)
        self.net_label_to_idx = lenet5_imagefolder.class_to_idx
        self.net_idx_to_label = dict([val,key] for key,val in self.net_label_to_idx.items())
        print("Digitnet loaddata class to idx:",self.net_label_to_idx)
        print("Digitnet loaddata idx to class:",self.net_idx_to_label)
        return
    def trainall(self,epochs,ui_lb=None, device = tf.device("cuda:0")): #cpu or cuda:0
        print("Digit trainall... device is:",  device)

        if self.lenet5_trainloader == None:
            print("Err digitnet:self.lenet5_trainloader == None")
            return 
        self.modelnet.to(device)
        loss_ep=0.0
        for ep in range(epochs):
            loss_ep = 0.0
            for i,data in enumerate(self.lenet5_trainloader):
                inputs,labels = data
                inputs,labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outs = self.modelnet(inputs)
                loss = self.lossfunc(outs, labels)   
                loss.backward()
                self.optimizer.step()
                loss_ep += loss.item()           
            print(f"{ep}/{epochs} loss:{loss_ep}")
        if ui_lb: ui_lb.setText(f"Train {epochs}, loss:{loss_ep}")
        return loss_ep
    def testall(self,device=tf.device("cuda:0")):
        global DIGIT_CLASSNUMS,DIGIT_TRAIN_BATCHSIZE
        global ui_idx_to_label
        print("Test for all... device is:",device)
        if self.lenet5_trainloader == None:
            print("Err digitnet:self.lenet5_trainloader == None")
            return
        self.modelnet.to(device)
        samples_all = 0
        samples_true = 0
        with tf.no_grad():
            for _,data in enumerate(self.lenet5_trainloader):
                inputs, labels = data #[10,1,38,38]  [10]
                inputs, labels = inputs.to(device), labels.to("cpu").squeeze()
                outs = self.modelnet(inputs)
                _,preds = tf.max(outs,1) #outs:10x18 pres:
                
                preds = preds.to("cpu").squeeze() #10:[1,12,17,0,1...]
                samples_all = samples_all + labels.numel()
                samples_true = samples_true + (labels == preds).squeeze().numpy().sum()
   
        print(f"Digit testall: {samples_true} corrects of {samples_all}.")     
        if samples_all <=0: return 0.0
        return samples_true/samples_all   
                #for i in range(DIGIT_CLASSNUMS):

    def loadnet(self):
        self.modelnet.load_state_dict(tf.load("./digitnet.pth", weights_only=True))
        print("Digitnet loadnet ok!")
        return
    def savenet(self):
        tf.save(self.modelnet.state_dict(),"./digitnet.pth")
        print("Save digitnet ok!")
        return 
        
if __name__ == "__main__":

    model = class_LeNet()

    ret = model(tf.randn(1,1,38,38))
    print("model ret shape:", ret.shape)
    lenet5_trans=transforms.Compose([
        transforms.Grayscale(),#此变换将以给定的概率水平（随机）翻转图像。通过参数“p”来设置这个概率。p的默认值为0.5。
        transforms.Resize([DIGIT_SIZEWH, DIGIT_SIZEWH]),
        transforms.ToTensor()
    ])
    lenet5_imagefolder = tfv.datasets.ImageFolder(DIGIT_IMG_ROOT, transform=lenet5_trans)
    lenet5_trainloader = tf.utils.data.DataLoader(lenet5_imagefolder,batch_size=10,shuffle=True)
    print("train_data classes:",lenet5_imagefolder.class_to_idx)
    # for i_batch, img in enumerate(lenet5_trainloader):
    #     if i_batch == 0:
    #         print('label:',img[1])
    #         fig = plt.figure()
    #         grid = tfv.utils.make_grid(img[0])
    #         print("shape 1:",img[1])
    #         plt.imshow(grid.numpy().transpose((1, 2, 0)))
    #         plt.show()
    #         #utils.save_image(grid,'test01.png')
    #     break
    titer = iter(lenet5_trainloader)
    imglb = next(titer)
   
    print(imglb[0][2].shape, imglb[1][2])
    plt.imshow(imglb[0][2].numpy().transpose((1, 2, 0)))
    plt.show()

    
