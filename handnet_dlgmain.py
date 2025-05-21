

import sys
import cv2
import random
from matplotlib import image
import numpy as np
import torch as tf
import torchvision as tfv
import torchvision.transforms as transforms

 
from PyQt5.QtWidgets import * #QWidget, QSizePolicy, QMainWindow, QApplication
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import Ui_dlg_handnet
from handnet_params_funcs import *
from handnet_linktrain import *
from handnet_digittrain import *

link_trainer = class_link_trainerEX()
digit_trainer = class_digit_trainer()

###global params...
MODE_NAMES={0:"link train",1:"digit train",2:"test"}

current_mode = 1
##link params
link_pts_seq=[]
link_pts_seqlist=[]  ##[[s1],[s2],[s3]] , s1:[timestart,timeend],[x0,y0],...,[xn-1,yn-1]
link_seqlist_bboxes=[]
link_seqlist_features5 = []
link_seqlist_labels = []
stn_calculator_result = ""


##digits params

digit_pts_seq = [] #like link_pts_seq [(10,10),(1,2)...]
digit_descriptor ={} #"lb","scale", "color", "box","seqs", "im","sel"
# {"lb":0,"scale":0.5,"color":(255,244,30),"box":(10,10,100,100),"seqs":[[ [1,1],[2,2] ],  [ [4,2],[10,3],[2,5] ]] } #pts:笔画数*笔画长度*2（点对）
digits_structs = [] #[digit_descriptor{},digit_descriptor,...]


image_main = np.zeros((1, 1,3),dtype="uint8")
image_sub = np.zeros((1, 1,3),dtype="uint8")
imagemain_size = (1,1)
imagesub_size = (1,1)

   
def calc_value(exp):
    try:
        v = eval(exp)
        print("Calc result:",v)
        return "Calc Result:"+str(v)
    except Exception as e:
        print("Calc Error!:",e)
        return "Calc Error!"

def check_comp_result():
    global  digit_descriptor,digits_structs
    global stn_calculator_result
    strs = ""
    stn_calculator_result = ""
    if len(digit_descriptor)<=0: return 
    if digit_descriptor["lb"] == 16:
        for desc in digits_structs:
            strs += ui_idx_to_comp[desc["lb"]]
        
        stn_calculator_result = calc_value(strs)
        
    return

def draw_image_sub(ui_lb):
    global imagesub_size, image_sub
    ##
    image_sub = np.zeros((imagesub_size[1], imagesub_size[0],3),dtype="uint8")
    if current_mode == 0:
        cv2.putText(image_sub,"Hello lb_stnlcd",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),1)
    elif current_mode==2:
        cv2.putText(image_sub,"Hello lb_stnlcd",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),1)
        cv2.putText(image_sub,"Calculator 1.0",(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
    show_image_in_lable(ui_lb,image_sub) 
    return
    

def draw_image_main(ui_lb):
    global imagemain_size, image_main
    global link_seqlist_bboxes, link_seqlist_labels,link_pts_seqlist, link_pts_seq
    global digit_pts_seq, digit_descriptor,digits_structs
    
    image_main = np.zeros((imagemain_size[1],imagemain_size[0],3),dtype="uint8")
    
    if current_mode == 0:
    ##draw points& lines
        draw_immain_seqlist(image_main,link_pts_seqlist)
        draw_immain_seq(image_main,link_pts_seq)
        cv2.putText(image_main,"Learn linknet",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        ##draw labels,boxes
        if len(link_seqlist_bboxes) != len(link_seqlist_labels):
            print("Err draw_image_main: len != len")
            return 
        for i in range(len(link_seqlist_bboxes)):
            b = link_seqlist_bboxes[i]   
            if link_seqlist_labels[i] == 0:
                cv2.putText(image_main,"No",(b[0],b[1]),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
                cv2.rectangle(image_main,b,(255,0,0),2)
            elif link_seqlist_labels[i] == 1:
                cv2.putText(image_main,"Yes",(b[0],b[1]),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                cv2.rectangle(image_main,b,(0,255,0),2)
            else:cv2.rectangle(image_main,b,(255,255,0),1)
    elif current_mode == 1: #global digit_pts_seq, digit_descriptor,digits_structs
        cv2.putText(image_main,"Learn digitnet",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        draw_immain_digitStructs(image_main,digits_structs)
        draw_immain_digitDescriptor(image_main,digit_descriptor)
        draw_immain_seq(image_main,digit_pts_seq)
        draw_immain_digitSelect(image_main, digits_structs)
    elif current_mode == 2:
        
        if stn_calculator_result:
            cv2.putText(image_main,stn_calculator_result,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        else:
            cv2.putText(image_main,"Calculator",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        draw_immain_digitStructs(image_main,digits_structs)
        draw_immain_digitDescriptor(image_main,digit_descriptor)
        draw_immain_seq(image_main,digit_pts_seq)
    show_image_in_lable(ui_lb, image_main)
    return

def reset_params():
    global link_pts_seqlist,link_pts_seq,link_seqlist_bboxes,link_seqlist_features5,link_seqlist_labels
    global digit_pts_seq, digit_descriptor,digits_structs
    global stn_calculator_result
    link_pts_seqlist=[]  ##[[s1],[s2],[s3]] , s1:[timestart,timeend],[x0,y0],...,[xn-1,yn-1]
    link_pts_seq=[]
    link_seqlist_bboxes=[]
    link_seqlist_features5 = []
    link_seqlist_labels = []
    
    digit_descriptor ={} # {"lb":0,"box":(10,10,100,100),"seqs":[[ [1,1],[2,2] ],  [ [4,2],[10,3],[2,5] ]] } #pts:笔画数*笔画长度*2（点对）
    digit_pts_seq = []
    digits_structs =[] 
    
    stn_calculator_result = ""
    return

def  reset_imgLBmain(ui_lb,sz):
    global imagemain_size, image_main
    imagemain_size= [sz.width(), sz.height()]
    image_main = np.zeros((imagemain_size[1],imagemain_size[0],3),dtype="uint8")
    draw_image_main(ui_lb)
    show_image_in_lable(ui_lb, image_main)
    return
    
def  reset_imgLBminishow(ui_lb,sz):
    global imagesub_size, image_sub 
    imagesub_size = [sz.width(), sz.height()]
    image_sub = np.zeros((imagesub_size[1], imagesub_size[0],3),dtype="uint8")
    draw_image_sub(ui_lb)
    show_image_in_lable(ui_lb,image_sub) 
    return
    
    
def lbmain_mouseMap(ctrl,srcpos):
    global imagemain_size
    mappos = [0,0]
    pt = ctrl.mapToGlobal(srcpos)
    ptLB = ctrl.lbim_main.mapToGlobal(QPoint(0,0))
    mappos[0] = pt.x() - ptLB.x()+1
    mappos[1] = pt.y() - ptLB.y()+1
    if mappos[0]<0: mappos[0]=-1
    if mappos[1]<0: mappos[0]=-1
    if mappos[0]>=imagemain_size[0]: mappos[0] = -1#self.immain_size[0]-1
    if mappos[1]>=imagemain_size[1]: mappos[1] = -1#self.immain_size[1]-1
    return mappos




def calc_update_digitDescriptor(seq, c= (255,255,255)):
    global DIGIT_DFT_SCALE,DIGIT_SIZEWH
    global digits_structs,digit_descriptor
    global digit_trainer

    if "seqs" not in digit_descriptor.keys(): digit_descriptor["seqs"]=[]
    if seq is not None:  digit_descriptor["seqs"].append(seq)
    digit_descriptor["color"] = c
    digit_descriptor["box"] = get_seqs_boundingbox(digit_descriptor["seqs"])
    digit_descriptor["sel"] = False
    
    S = 0
    for d in digits_structs:
        s = max(d["box"][2],d["box"][3])
        S = S + s
    N = len(digits_structs)
    
    if N<=0: 
        digit_descriptor["scale"] = DIGIT_DFT_SCALE
    else:
        avgs = S/N
        s = max(digit_descriptor["box"][2], digit_descriptor["box"][3])
        r = s/avgs
        if r>0.95: r=1.0
        if r<0.25: r=0.25
        digit_descriptor["scale"] = DIGIT_DFT_SCALE*r
        
    digit_descriptor["im"]=digit_DescToImage(digit_descriptor,DIGIT_SIZEWH,digit_descriptor["scale"] ) 
    digit_descriptor["lb"] = -1
    digit_predictFromDesc(digit_trainer,digit_descriptor)  
    return 

def calc_update_digitsStruct():
    global DIGIT_DFT_SCALE,DIGIT_SIZEWH
    global digits_structs
    global digit_trainer
    dstruct = digits_structs
    if len(dstruct)<=0: return 
    S = 0
    for desc in dstruct:   S = S +  max(desc["box"][2],desc["box"][3])
    savg = S/len(dstruct)
    #print("Digits update...")
    for i in range(len(dstruct)):
        s = max(dstruct[i]["box"][2], dstruct[i]["box"][3])
        r = s/savg
        if r>0.95: r=1.0
        if r<0.25: r=0.25
        
        newscale = DIGIT_DFT_SCALE * r
        if abs(newscale - dstruct[i]["scale"])> 0.15:
            dstruct[i]["scale"]  = newscale
            dstruct[i]["im"]=digit_DescToImage(dstruct[i],DIGIT_SIZEWH,newscale)
            dstruct[i]["lb"] = -1
            digit_predictFromDesc(digit_trainer,dstruct[i])
            print("Digits update Scale:", i)
        if dstruct[i]["lb"] < 0:
            digit_predictFromDesc(digit_trainer,dstruct[i])
    return
    

############################
class Dlg_main(QDialog, Ui_dlg_handnet.Ui_dlg_main):
    def __init__(self, parent = None):
        global link_trainer
        super(Dlg_main, self).__init__(parent)
        self.setupUi(self)
        ####################init&set UI...
        self.setWindowFlag(Qt.WindowMinMaxButtonsHint)
        #self.setWindowState(Qt.WindowMaximized)
        self.setWindowTitle("HandCalc. by stnlcd(2022.2.5)")    
        self.lb_info.setText("laobei250") #debug
        self.lbim_main.move(1,1)
        self.lbim_minishow.move(1,1)
        self.lbim_main.setScaledContents(True)
        self.lbim_minishow.setScaledContents(True)
        self.cb_mode_select.addItem("Link Train")
        self.cb_mode_select.addItem("Digit Train")
        self.cb_mode_select.addItem("Calc Test")
        self.cb_mode_select.setCurrentIndex(current_mode)
        self.stk_widget.setCurrentIndex(current_mode)
        self.cb_mode_select.currentIndexChanged.connect(self._on_cb_mode_index_changed)
        show_linklearn_info(self.lb_linklearn_info, link_trainer)
        self.bt_link_learnnew.clicked.connect(self.on_link_learnnew)
        self.bt_link_savenet.clicked.connect(self.on_bt_link_savenet)
        self.bt_link_addnew_csv.clicked.connect(self.on_bt_link_addcsv)
        self.bt_link_learnall.clicked.connect(self.on_bt_link_learnall)
        self.bt_link_testall.clicked.connect(self.on_link_testall)
        self.bt_link_testnew.clicked.connect(self.on_link_testnew)
        self.bt_classradios = [self.rb_tr_0,self.rb_tr_1,self.rb_tr_2,self.rb_tr_3,self.rb_tr_4,self.rb_tr_5,
                               self.rb_tr_6,self.rb_tr_7,self.rb_tr_8,self.rb_tr_9,self.rb_tr_add,
                               self.rb_tr_sub,self.rb_tr_mul,self.rb_tr_div, self.rb_tr_dot,
                               self.rb_tr_leftk, self.rb_tr_eq, self.rb_tr_rightk]
        
        for e in self.bt_classradios:
            e.setCheckable(True)
            e.toggled.connect(self.on_bt_classRadios)
        self.bt_digit_saveimgs.clicked.connect(self.on_bt_digit_saveimgs)
        self.bt_digit_learnall.clicked.connect(self.on_bt_digit_learnall)
        self.bt_digit_savenet.clicked.connect(self.on_bt_digit_savenet)
        self.bt_digit_testall.clicked.connect(self.on_bt_digit_testall)
        self.bt_digit_testnew.clicked.connect(self.on_bt_digit_testnew)
        self.bt_calc_back.clicked.connect(self.calc_back)
        self.bt_calc_clear.clicked.connect(self.on_bt_calc_clear)
        show_imagefiles_num(self.lb_info,get_imagefiles_count())
        
        #########params
        self.left_btn_state = 0
        self.mouse_position=[-1,-1] #[x,y]

        reset_imgLBmain(self.lbim_main, self.widget_lbmain.size())
        reset_imgLBminishow(self.lbim_minishow,self.widget_lbmini.size())
        return
    def on_bt_digit_testnew(self):
        global digits_structs
        global digit_trainer
        for desc in digits_structs:
            desc["lb"] = -1
            digit_predictFromDesc( digit_trainer,desc)
        show_labeledinfo(self.lb_digit_info,digits_structs)
        draw_image_main(self.lbim_main) 
        return 
    def on_bt_digit_testall(self):
        global digit_trainer
        self.lb_digit_info.setText("Testing all...")
        msgb = QMessageBox.question(self,"Digit test.","Test for all?",QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
        if msgb == QMessageBox.No: return
        else: print("Ignore!")

        digit_trainer.loaddata()
        digit_trainer.loadnet()
        accu=digit_trainer.testall()
        strs = f"Testall accuracy:%{accu*100}"
        print(strs)
        self.lb_digit_info.setText(strs)
        return
    def on_bt_digit_savenet(self):
        global digit_trainer
        msgb = QMessageBox.question(self,"Save lenet.","Save lenet?",QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
        if msgb == QMessageBox.Yes:
            digit_trainer.savenet()
        else: print("Ignore!")
        return 
    def on_bt_digit_learnall(self):
        global digit_trainer
        self.lb_digit_info.setText("Learning all...")
        msgb = QMessageBox.question(self,"Digit learn.","Learn for all?",QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
        if msgb == QMessageBox.No: return
        digit_trainer.loaddata()
        digit_trainer.loadnet()
        loss = digit_trainer.trainall(self.SB_digit_allepoch.value(),self.lb_info) #SB_digit_allepoch
        self.lb_digit_info.setText("Finished. with loss:%.4f"%(loss))
        msgb = QMessageBox.question(self,"Train finished","Save lenet?",QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
        if msgb == QMessageBox.Yes:
            digit_trainer.savenet()
        else: print("Ignore!")   
        return
    def on_bt_digit_saveimgs(self):
        global digits_structs

        msgb = QMessageBox.question(self,"Save images","Save new images?",QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
        if msgb == QMessageBox.Yes:
            self.lb_digit_info.setText("Saving images...")
            
            show_imagefiles_num(self.lb_info,save_new_digits(digits_structs))
            self.lb_digit_info.setText("Images saved...")
        else:
            print("Ignore!")
        return
    def on_bt_classRadios(self):
        global digits_structs
        if self.sender().isChecked() == False: return 
        for desc in digits_structs:
            if desc["sel"] == False: continue
            #id = self.bt_classradios.index(self.sender())
            #cl = self.ui_idx_to_label[id]
            desc["lb"] = self.bt_classradios.index(self.sender())
        show_labeledinfo(self.lb_digit_info,digits_structs)
        draw_image_main(self.lbim_main) 
        #print("clk:", cl)
        return 

    def on_bt_link_addcsv(self):
        global link_seqlist_features5, link_seqlist_labels
        msgb = QMessageBox.question(self,"Save to csv?","Save new datas to csv?",QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
        if msgb == QMessageBox.Yes:
            link_trainer.save_new_csv(link_seqlist_features5,link_seqlist_labels)
            show_linklearn_info(self.lb_linklearn_info,link_trainer)
        else:
            print("Ignore!")
        return
    def on_bt_link_savenet(self):
        msgb = QMessageBox.question(self,"Save Net?","Save the link net?",QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
        if msgb == QMessageBox.Yes:
            link_trainer.save_net()
        else:
            print("Ignore!")
        return
            
    def on_link_testnew(self):
        global link_seqlist_features5, link_seqlist_labels
        self.lb_info.setText("Testing new ...")
        link_trainer.get_new_accuracy(link_seqlist_features5, link_seqlist_labels)
        self.lb_info.setText("Test new finish!")
        return
    def on_link_testall(self):
        self.lb_info.setText("Testing all ...")
        link_trainer.get_all_accuracy()
        self.lb_info.setText("Test all finish!")
        return
    def on_bt_link_learnall(self):
        self.lb_info.setText("Learning all ...")
        epochs = self.SB_link_allepoch.value()
        link_trainer.train_all(epochs)
        self.lb_info.setText("Learn all finish!")
        return
    def on_link_learnnew(self):
        global link_seqlist_features5, link_seqlist_labels
        self.lb_info.setText("Learning new ...")
        epochs = self.SB_link_newepoch.value()
        link_trainer.train_new(link_seqlist_features5,link_seqlist_labels,epochs)
        self.lb_info.setText("Learn neww finish!")
        return
    def _on_cb_mode_index_changed(self,i):
        global current_mode
        current_mode = i
        self.lb_info.setText(f"Change mode to {MODE_NAMES[i]}")
        print(f"Change mode to {MODE_NAMES[i]}")
        self.stk_widget.setCurrentIndex(i)
        reset_params()  
        reset_imgLBmain(self.lbim_main, self.widget_lbmain.size())
        reset_imgLBminishow(self.lbim_minishow,self.widget_lbmini.size())
        return
        
    def resizeEvent(self, a0: QResizeEvent):
        global imagemain_size, imagesub_size

        imagemain_size = [self.widget_lbmain.size().width(), self.widget_lbmain.size().height()]
        self.lbim_main.resize(self.widget_lbmain.size())
        
        imagesub_size= [self.widget_lbmini.size().width(), self.widget_lbmini.size().height()]
        self.lbim_minishow.resize(self.widget_lbmini.size())
          
        reset_params()  
        reset_imgLBmain(self.lbim_main, self.widget_lbmain.size())
        reset_imgLBminishow(self.lbim_minishow,self.widget_lbmini.size())
        
        print("window size changed:",imagemain_size)  
        return super().resizeEvent(a0) 
        return
    def mouseDoubleClickEvent(self,e:QMouseEvent):
        if current_mode == 0: self.mouseDoubleClickEvent_mode0(e)
        elif current_mode == 1: self.mouseDoubleClickEvent_mode1(e)
    def mousePressEvent(self,e: QMouseEvent):
        if current_mode == 0: self.mousePressEvent_mode0(e)
        elif current_mode==1: self.mousePressEvent_mode1(e)
        elif current_mode==2: self.mousePressEvent_mode2(e)
    def mouseReleaseEvent(self, e: QMouseEvent):
        if current_mode==0: self.mouseReleaseEvent_mode0(e)
        elif current_mode==1: self.mouseReleaseEvent_mode2(e)
        elif current_mode==2: self.mouseReleaseEvent_mode2(e)
    def mouseMoveEvent(self, e: QMouseEvent):
        if current_mode==0: self.mouseMoveEvent_mode0(e)
        elif current_mode==1: self.mouseMoveEvent_mode1(e)
        elif current_mode==2: self.mouseMoveEvent_mode1(e)
        
    def mouseDoubleClickEvent_mode1(self,e:QMouseEvent):  
        global digits_structs
        if e.button() == Qt.LeftButton:
            reset_params()
            reset_imgLBmain(self.lbim_main,self.widget_lbmain.size())
            reset_imgLBminishow(self.lbim_minishow, self.widget_lbmini.size())
            show_labeledinfo(self.lb_digit_info,digits_structs)
        elif e.button() == Qt.RightButton:
            for desc in digits_structs:
                desc["lb"] = -1
            if "lb" in digit_descriptor.keys():
                digit_descriptor["lb"] = -1
            show_labeledinfo(self.lb_digit_info,digits_structs)
            draw_image_main(self.lbim_main)
 
    def mouseDoubleClickEvent_mode0(self,e:QMouseEvent):
        if e.button() == Qt.LeftButton:
            reset_params()
            reset_imgLBmain(self.lbim_main,self.widget_lbmain.size())
            reset_imgLBminishow(self.lbim_minishow, self.widget_lbmini.size())

            
    def mousePressEvent_mode0(self,e: QMouseEvent):
        global link_pts_seqlist,link_pts_seq,link_seqlist_bboxes,link_seqlist_features5, link_seqlist_labels
        self.mouse_position = lbmain_mouseMap(self,e.pos())
        
        for i in range(len(link_seqlist_bboxes)):
            if point_in_rect(link_seqlist_bboxes[i], self.mouse_position)>0:
                if e.button() == Qt.LeftButton:
                    link_seqlist_labels[i] = 1
                elif e.button() ==Qt.RightButton:
                    link_seqlist_labels[i] = 0
                
                reset_imgLBmain(self.lbim_main,self.widget_lbmain.size())
                #show_infostatus_f5(self.lb_info,link_seqlist_features5[i],link_seqlist_labels[i] )
                draw_image_main(self.lbim_main)
                self.left_btn_state = 2
                return
                    
        if e.button() == Qt.LeftButton: 
            self.left_btn_state = 1
            if self.mouse_position[0]<0 or self.mouse_position[1]<0: return
            link_pts_seq=[]
            link_pts_seq.append(self.mouse_position)    
        elif e.button() == Qt.RightButton:
            print("right click")
            return
        draw_image_main(self.lbim_main)
        
    def on_bt_calc_clear(self):
        reset_params()
        reset_imgLBmain(self.lbim_main, self.widget_lbmain.size())
        reset_imgLBminishow(self.lbim_minishow,self.widget_lbmini.size())
    def calc_back(self):
        global digit_descriptor,digits_structs,digit_pts_seq
        if current_mode!=2: return
        digit_pts_seq = []
        if len(digit_descriptor)>0: digit_descriptor = {}
        elif len(digits_structs)>0:   digits_structs.pop()
        draw_image_main(self.lbim_main)

        return
    def mousePressEvent_mode2(self,e: QMouseEvent): ##calc
        global digit_descriptor,digits_structs,digit_pts_seq
        global image_sub
        self.mouse_position = lbmain_mouseMap(self,e.pos())
        if e.button() == Qt.RightButton:
            self.calc_back()
        elif e.button() == Qt.LeftButton: 
            self.left_btn_state = 1
            if self.mouse_position[0]<0 or self.mouse_position[1]<0: return
            digit_pts_seq=[]
            digit_pts_seq.append(self.mouse_position)  
        draw_image_main(self.lbim_main)
        return
    def mousePressEvent_mode1(self,e: QMouseEvent):
        global digit_descriptor,digits_structs,digit_pts_seq
        global image_sub
        global digit_trainer
        self.mouse_position = lbmain_mouseMap(self,e.pos())
        if e.button() == Qt.RightButton:
            unlabel_selected_digit(self.mouse_position,digits_structs)
            show_labeledinfo(self.lb_digit_info,digits_structs)
        elif e.button() == Qt.LeftButton: 
            desc = check_digits_mouseclk(self.mouse_position,digits_structs)     
            if desc !=None:  #click in 
                digit_predictFromDesc(  digit_trainer,desc)
                show_labeledinfo(self.lb_digit_info,digits_structs)
                draw_digit_sub(self.lbim_minishow,image_sub, digits_structs)
                if len(digit_descriptor)>0:
                    c = (random.randint(60,255),random.randint(60,255),random.randint(60,255))
                    #set_digit_descriptor(digit_descriptor,-1,DIGIT_DFT_SCALE,c,None)
                    calc_update_digitDescriptor(None,c)
                    digits_structs.append(digit_descriptor)
                    
                    #update_digits_scale(digits_structs)
                    calc_update_digitsStruct()
                    digit_descriptor = {}
                    digit_pts_seq = []
                draw_image_main(self.lbim_main)
                
                for e in self.bt_classradios: 
                    e.setCheckable(False)
                    e.setCheckable(True)
                self.update()
                return
            self.left_btn_state = 1
            if self.mouse_position[0]<0 or self.mouse_position[1]<0: return
            digit_pts_seq=[]
            digit_pts_seq.append(self.mouse_position)  
        draw_image_main(self.lbim_main)
        return
    def mouseMoveEvent_mode0(self, e: QMouseEvent):
        global link_pts_seq, link_pts_seqlist
        if self.left_btn_state != 1:return
        self.mouse_position = lbmain_mouseMap(self,e.pos())
        if self.mouse_position[0]<0 or self.mouse_position[1]<0: return
        link_pts_seq.append(self.mouse_position)
        draw_image_main(self.lbim_main)
        return 
    def mouseMoveEvent_mode1(self, e: QMouseEvent):
        global digit_pts_seq
        if self.left_btn_state != 1:return
        self.mouse_position = lbmain_mouseMap(self,e.pos())
        if self.mouse_position[0]<0 or self.mouse_position[1]<0: return
        digit_pts_seq.append(self.mouse_position)
        draw_image_main(self.lbim_main)
        return
    def mouseReleaseEvent_mode2(self,e: QMouseEvent):
        global DIGIT_DFT_SCALE
        global digit_pts_seq, digit_descriptor,digits_structs
        if e.button() != Qt.LeftButton: return
        if self.left_btn_state!= 1: return
        self.left_btn_state = 0
        if len(digit_pts_seq) <= 0: return
        
        if len(digit_descriptor)<=0:
  
            calc_update_digitDescriptor(digit_pts_seq)
            calc_update_digitsStruct()
            digit_pts_seq = []    
            draw_image_main(self.lbim_main) 
            return
        predy = link_trainer.check_merge(digit_descriptor["seqs"][-1], digit_pts_seq)
        if predy==1: #merge
            calc_update_digitDescriptor(digit_pts_seq)
            calc_update_digitsStruct()
            if current_mode==2:check_comp_result()
        else:

            digit_descriptor["color"] = (random.randint(60,255),random.randint(60,255),random.randint(60,255))    
            digits_structs.append(digit_descriptor)

            digit_descriptor = {}
            calc_update_digitDescriptor(digit_pts_seq)
            calc_update_digitsStruct()
        digit_pts_seq = []

        draw_image_main(self.lbim_main) 
        return
   
    def mouseReleaseEvent_mode0(self,e: QMouseEvent):
        global link_pts_seq, link_pts_seqlist, image_main
        if self.left_btn_state==2:
            self.left_btn_state = 0
            return
        if e.button() != Qt.LeftButton: return
        self.left_btn_state = 0
        if len(link_pts_seq)>0: link_pts_seqlist.append(link_pts_seq)
        link_pts_seq = []
        #print("released link_pts_seqlist len:", len(self.link_pts_seqlist))
        if  len(link_pts_seqlist)>0 and len(link_pts_seqlist)%2==0 :
            link_seqlist_bboxes.append( get_seqs_boundingbox(link_pts_seqlist[-2:]) )
            link_seqlist_labels.append(-1)     
            link_seqlist_features5.append( get_seqs2_featuresEX(link_pts_seqlist[-2:]) )
              
        draw_image_main(self.lbim_main)
        return 



###main
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainw = Dlg_main()
    mainw.show()
    sys.exit(app.exec_())