

import cv2 
from PyQt5.QtWidgets import * #QWidget, QSizePolicy, QMainWindow, QApplication
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
from handnet_linktrain import *

ui_idx_to_label={0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",
                            10:"add",11:"sub",12:"mul",13:"div",14:"dot",15:"leftk",16:"eq",17:"rightk"}
ui_idx_to_comp ={0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",
                            10:"+",11:"-",12:"*",13:"/",14:".",15:"(",16:"=",17:")"}
ui_label_to_idx = {"0":0, "1":1,"2":2, "3":3, "4":4,"5":5,"6":6, "7":7, "8":8, "9":9,"0":0,
                         "add":10, "sub":11, "mul":12, "div":13, "dot":14, "leftk":15, "eq":16, "rightk":17}

def show_linklearn_info(ui_lb, trainer):
     str = "%dP-%dN"%(trainer.link_trains_pN,trainer.link_trains_nN)
     ui_lb.setText(str)
     return 


def draw_digit_sub(ui_lb, img, dstruct):
    imsel = None
    for desc in dstruct:
        if desc["sel"] == True:
            imsel = desc["im"]
            #print("scale:",desc["scale"])
            break
    if imsel is None: return
    wh = min((img.shape[0],img.shape[1]))
    imsel = cv2.resize(imsel,(wh,wh))
    show_image_in_lable(ui_lb,imsel) 
    #cv2.imshow("imsel",imsel)
    return 

def show_image_in_lable(ui_lb,img):
    qimg = QImage(img.data,img.shape[1],img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
    ui_lb.setPixmap(QPixmap.fromImage(qimg))
    ui_lb.adjustSize()

def point_in_rect(rect,pt): #(10,10,100,100)  (40,40) True
    nprect = np.array([[rect[0],rect[1]], 
                       [rect[0]+rect[2],rect[1]],
                       [rect[0]+rect[2],rect[1]+rect[3]],
                       [rect[0],rect[1]+rect[3]]])
    nprect = nprect.reshape([4, 1, 2]).astype(np.int64)
    return cv2.pointPolygonTest(nprect,tuple(pt),False)

def get_seqs_boundingbox(seqs):#[[(1,2),(23,12)],[(22,11),(12,11)]]
    pts = []
    for seq in seqs:
        for e in seq:
            pts.append(e)
    box = list(cv2.boundingRect(np.array(pts)))
    
    box[2] = max(box[2],3)
    box[3] = max(box[3],3)
    return tuple(box)

def show_infostatus_f5(ui_lb,f5,lb):
    #'hratio','wratio','sin','cos','andor'
    str = "[H:%.2f W:%.2f sin:%.2f cos:%.2f andor:%.2f label:%d]"%(f5[0],f5[1],f5[2],f5[3],f5[4],int(lb))
    ui_lb.setText(str)
    
    
def draw_immain_seq(img,seq,color=(255,255,255),R=2):

    for i in range(len(seq)):
        if seq[i][0]<0 or seq[i][1]<0: continue
        cv2.circle(img,tuple(seq[i]),R,color,-1)
    for i in range(len(seq)-1):
        if seq[i][0]<0 or seq[i][1]<0: continue
        if seq[i+1][0]<0 or seq[i+1][1]<0: continue
        cv2.line(img,tuple(seq[i]),tuple(seq[i+1]),color,R+1)
    return 
        
def draw_immain_seqlist(img, seqlist, color=(255,255,255),R=2):

    for seq in seqlist: draw_immain_seq(img,seq,color,R)
    return 

def draw_immain_digitDescriptor(img, desc, color=(255,255,255),R=2):

    if "color" in desc.keys(): color = desc["color"]
    if "seqs" in desc.keys():
        draw_immain_seqlist(img, desc["seqs"],color,R)

    if "box" in desc.keys():
        box = desc["box"]
        if desc["lb"]<0: cv2.rectangle(img,box,(255,255,0),1)
        else:
            cv2.rectangle(img, box, (0,255,0),1)
            lb = ui_idx_to_label[desc["lb"]]
            cv2.putText(img,lb,(box[0],box[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1) 
            
    return

def draw_immain_digitStructs(img, dstruct):
    for desc in dstruct:
        draw_immain_digitDescriptor(img, desc)
#cv2.putText(image_main,"hello lb250!",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)   
    return     
def draw_immain_digitSelect(img,dstruct):
    global ui_idx_to_label
    for desc in dstruct:
        box = desc["box"]
        if desc["sel"] == True: cv2.rectangle(img,box,(205,205,205),3) 
    return


        
                





