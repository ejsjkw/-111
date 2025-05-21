import cv2
import numpy as np
import pandas as pd 
import torch as tf
import torch.nn as nn
import torch.optim as optim

from PyQt5.QtWidgets import * #QWidget, QSizePolicy, QMainWindow, QApplication
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from handnet_params_funcs import *

def rect_or(a,b):
    x = min(a[0],b[0])
    y = min(a[1],b[1])
    w = max(a[0]+a[2],b[0]+b[2])-x
    h = max(a[1]+a[3],b[1]+b[3])-y
    return (x,y,w,h)

def rect_and(a,b):
    x = max(a[0],b[0])
    y = max(a[1],b[1])
    w = min(a[0]+a[2],b[0]+b[2])-x
    h = min(a[1]+a[3],b[1]+b[3])-y
    if w<0 or h<0: return (0,0,0,0)
    return (x,y,w,h)

def rect_andDIVor(a,b):
    #andab = rect_and(a,b) #a[2]*a[3] + b[2]*b[3] #
    #sand = andab[2]*andab[3]
    sand = a[2]*a[3] + b[2]*b[3] 
    orab = rect_or(a,b)
    sor = orab[2]*orab[3]*2.0+0.5
    return sand/sor

def seq_center(seq):#[(1,2),(23,12)]
    rnp = np.array(seq)
    return ( np.mean(rnp[:,0]),  np.mean(rnp[:,1]))
    

def vec2_sin_cos(v1,v2): #input:(10,2),(-4,8)
    v1 = np.array(v1,dtype=np.int32)
    v2 = np.array(v2,dtype=np.int32)
    v = v2-v1
    v22 = np.sum(v**2)
    d = np.sqrt(v22)
    if d<=3.0: return (1, 0, 0)
    return (d, (v[0]/d+1.0)/2.0, (v[1]/d+1.0)/2.0)

def get_seqs2_featuresEX(seqs):#[[(1,2),(23,12)],[(22,11),(12,11)]]
    if len(seqs)!=2:
        print("err get_seqs2_featuresEX: len(seqs)!=2")
        return
    if len(seqs[0])<=0 or len(seqs[1])<=0:
        print("err get_seqs2_featuresEX: len(seq)<=0")
        return 
    #seqs12 =  np.vstack((np.array(seqs[0],seqs[1])))
    #cm12 = seq_center(seqs12)
    c1,c2 = seq_center(seqs[0]),seq_center(seqs[1])
    r1 = cv2.boundingRect(np.array(seqs[0]))
    r2 = cv2.boundingRect(np.array(seqs[1]))
    er1 = cv2.minAreaRect(np.array(seqs[0]))  #((center1,center2),(len1,len2),ang)
    er2 = cv2.minAreaRect(np.array(seqs[1]))
    sq1 = len(seqs[0])
    sq2 = len(seqs[1])
    
    dc1c2,f1,f2 = vec2_sin_cos(c1,c2)
    f3 = min(sq1,sq2)/(max(sq1,sq2)+0.1)
    f4 = r1[2]/(r1[2]+r2[2]+1.0) #min(r1[2],r2[2])/(max(r1[2],r2[2])+0.1)
    f5 = r1[3]/(r1[3]+r2[3]+1.0) #min(r1[3],r2[3])/(max(r1[3],r2[3])+0.1)
    f6 = rect_andDIVor(r1,r2)
    f7 = dc1c2/( r1[2]+r2[2]+0.1) # er1[1][0]/dc1c2
    f7 = min(2.0, f7)
    f8 = dc1c2/(r1[3]+r2[3]+0.1)
    f8 = min(2.0,f8)
    
    f9= (er1[1][0]+er1[1][1])/(2*dc1c2+0.1)
    f9 = min(2.0,f9)
    f10=(er2[1][0]+er2[1][1])/(2*dc1c2+0.1)
    f10 = min(2.0,f10)
    

    fs = (f1,f2,f3,f4,f5,f6,f7,f8,f9,f10) 
    #print(f"sin:{f1} cos:{f2} s-S:{f3} w-W:{f4} h-H:{f5} andor:{f6} dh:{f7} dw:{f8} deh:{f9} dew:{f10}")
    return fs
    


######################


class class_link_netEX(nn.Module):
    def __init__(self):
        super(class_link_netEX,self).__init__() 
        self.l1 = nn.Linear(10,40)
        self.l2 = nn.Linear(40,2)
        self.cri = nn.CrossEntropyLoss()  #[0.223,0.11,0.444] 与 [2]比较
        return
    def forward(self,x):
        x = self.l1(x)
        x = tf.tanh(x)
        x = self.l2(x)
        return x
    def predict(self,x): #x:(N, FS) FS:feature size
        pred = tf.softmax(self.forward(x),dim=1)
        return tf.argmax(pred,dim=1)
    def getloss(self,x,y):
        y_pred = self.forward(x)
        return self.cri(y_pred,y)
        

class class_link_trainerEX():
    def __init__(self,devicestr= "cpu"):  #"cuda:0"  or "cpu"
        self.link_csv = pd.read_csv('./link_trainEx.csv')
        self.link_trains = np.array(self.link_csv[["f1","f2","f3","f4","f5","f6","f7","f8","f9","f10"]])
        self.link_labels = np.array(self.link_csv['lb'], dtype=np.int64)  # 将np.long改为np.int64
        self.link_trains_pN = np.sum(self.link_labels>0.5)
        self.link_trains_nN = np.sum(self.link_labels<0.5)


        print("link netEx device:",devicestr)
        #########
        self.device = tf.device(devicestr)
        self.modelnet = class_link_netEX()
        self.modelnet.to(self.device )
        self.lossfunc = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.modelnet.parameters(), lr=0.01, momentum=0.9)
        
        self.load_net()
        return

    def save_new_csv(self,f5,label):
        lbnp = np.array(label,dtype=np.long)
        lbflag = np.argwhere(lbnp<0) 
        lbnp = np.delete(lbnp,lbflag)
        if lbnp.size <=0 :
            print("No new data to write.")
            return
        f5np = np.array(f5,dtype=np.float32)
        f5np = np.delete(f5np,lbflag,axis=0)
     
        self.link_trains = np.vstack((self.link_trains,f5np))
        self.link_labels = np.hstack((self.link_labels,lbnp))
        
        self.link_trains_pN = np.sum(self.link_labels>0.5)
        self.link_trains_nN = np.sum(self.link_labels<0.5)
        
        link_numpy = np.concatenate((self.link_trains,self.link_labels.reshape(-1,1)),axis=1) 
        link_pad = pd.DataFrame(link_numpy,columns=self.link_csv.columns,index=None)
        link_pad.to_csv("link_trainEx.csv",index=False)  #将数据帧保存到csv文件
        print("CSV saved!")
        return
        
    def save_net(self):
        tf.save(self.modelnet.state_dict(),"./linknetEx.pth")
        print("linknetEx saved!")
        return
    def load_net(self):
        self.modelnet.load_state_dict(tf.load("./linknetEx.pth", weights_only=True))
        return
    def trainnet(self,f5tf, labeltf,epochs):
        for ep in range(epochs):
            out = self.modelnet(f5tf)
            loss = self.lossfunc(out, labeltf)
            self.optimizer .zero_grad()
            loss.backward()
            self.optimizer.step()
            with tf.no_grad():
                lossv = loss.item()
                print(f"{ep}/{epochs} loss:{lossv}")
        return
    def train_new(self,f5,label,epochs):
        print("Train new...")
        lbnp = np.array(label,dtype=np.long)
        lbflag = np.argwhere(lbnp<0)
        lbnp = np.delete(lbnp,lbflag)
        if lbnp.size <=0 :
            print("No data to train.")
            return

        f5np = np.array(f5,dtype=np.float32)
        f5np = np.delete(f5np,lbflag,axis=0)
        f5tf = tf.tensor(f5np,dtype=tf.float32).to(self.device)
        labeltf = tf.tensor(lbnp,dtype=tf.long).to(self.device)
        self.trainnet(f5tf,labeltf,epochs)
        return 
    def train_all(self,epochs):
        print("Train all...")
        f5tf = tf.tensor(self.link_trains,dtype=tf.float32).to(self.device)
        labeltf = tf.tensor(self.link_labels,dtype=tf.long).to(self.device)
        self.trainnet(f5tf,labeltf,epochs)
        return
    def get_new_accuracy(self,f5,label):
        print("Test for new...")
        lbnp = np.array(label,dtype=np.long)
        lbflag = np.argwhere(lbnp<0)
        lbnp = np.delete(lbnp,lbflag)
        if lbnp.size <=0 :
            print("No data to train.")
            return

        f5np = np.array(f5,dtype=np.float32)
        f5np = np.delete(f5np,lbflag,axis=0)
        f5tf = tf.tensor(f5np,dtype=tf.float32).to(self.device)
        predy = self.modelnet.predict(f5tf)
        predy = predy.cpu().numpy().astype(np.int64)  # 将np.long改为np.int64
        
        num_correct = (predy == lbnp).sum()
        accu = num_correct / predy.shape[0]
        
        print("new accuracy:",accu)
        return
    def get_all_accuracy(self):
        print("Test for all...")
        f5tf = tf.tensor(self.link_trains,dtype=tf.float32).to(self.device)

        predy = self.modelnet.predict(f5tf)
        predy = predy.cpu().numpy().astype(np.long)
        
        num_correct = (predy == self.link_labels).sum()
        accu = num_correct / predy.shape[0]
    
        print("all accuracy:",accu)
        return 
    def check_merge(self,seq1,seq2):

        seqs = []
        seqs.append(seq1)
        seqs.append(seq2)

        if len(seqs)!=2:
            print("Err check_merge:",len(seqs)!=2 )
            return 0
        f5 =[]
        f5.append( get_seqs2_featuresEX(seqs))
        f5tf = tf.tensor(f5,dtype=tf.float32).to(self.device)
        #print("f5tf:",f5tf)
        predy = self.modelnet.predict(f5tf)
        predy = predy.cpu().numpy().astype(np.int64)
        #print("check_merge predy:",predy[0])
        return predy[0]







# class linktrain_thread(QThread):
#     sig_train_finish = pyqtSignal()
#     def __init__(self, parent=None):
#         super(linktrain_thread,self).__init__(parent)
    
#     def run(self):
#         for i in range(10):
#             print("linktrain_thread run..")
#             self.sleep(0.5)
        
        

if __name__ == "__main__":

    
    # model = class_link_netEX()
    # tf.save(model.state_dict(),"./linknetEx.pth")
    # print("linknetEx saved!")
    pass