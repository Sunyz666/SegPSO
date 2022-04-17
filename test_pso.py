# encoding:utf-8
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
#import data_for_unet
from collections import OrderedDict
import os
import torch.backends.cudnn as cudnn
import math
from PIL import Image
import numpy as np
import cv2
import configparser
import matplotlib.pyplot as plt

from util import tensor2im, creatContour, contour2LightMaskImage, basedGeometry2Contour, dl_score_with_lightmask,\
                 get_img_contours, dl_score_with_lightmask_test,get_contours_from_image
from pso_x import pso

import multiprocessing
import time
from dataset import REFUGE
from modeling import models_v1 as models
import random
from loss import diceCoeff,SegmentationLosses
import tqdm
import dataPSOTest
class diceCoeff(nn.Module):
    def __init__(self):
        super(diceCoeff,self).__init__()
        
    def forward(self,pred, gt, smooth=1e-5, activation='sigmoid'):
        if activation is None or activation == "none":
            activation_fn = lambda x: x
        elif activation == "sigmoid":
            activation_fn = nn.Sigmoid()
        elif activation == "softmax2d":
            activation_fn = nn.Softmax2d()
        else:
            raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")

        pred = activation_fn(pred)

        N = gt.size(0)
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)
        
        intersection = (pred_flat * gt_flat).sum(1)
        unionset = pred_flat.sum(1) + gt_flat.sum(1)
        loss = (2 * intersection + smooth) / (unionset + smooth)
        return loss.sum()/N

    
def cal_score(oldcontours,outputcontours):
    emptyImage = np.zeros((256,256,3),np.uint8)
    emptyImage[:,:,:] = 255

    produceImage = emptyImage.copy()
    cv2.drawContours(produceImage, oldcontours, -1, (0, 0, 0), -1) # 旧边界图

    produceImageBifur = emptyImage.copy()
    cv2.drawContours(produceImageBifur, outputcontours, -1, (0, 0, 0), -1) # 新边界图

    GT_image =  cv2.cvtColor(produceImage, cv2.COLOR_BGR2RGB)
    GT_image[np.where(GT_image<255)] = 1
    GT_image[np.where(GT_image==255)] = 0
    GT_image_01 = GT_image[:,:,0]

    Bif_image =  cv2.cvtColor(produceImageBifur, cv2.COLOR_BGR2RGB)
    Bif_image[np.where(Bif_image<255)] = 1
    Bif_image[np.where(Bif_image==255)] = 0
    Bif_image_01 = Bif_image[:,:,0]


    mix_image = GT_image_01 + Bif_image_01
    
    eps = 1e-7
    merge  = np.where(mix_image==2)[0].shape[0]
    combine = np.where(mix_image>0)[0].shape[0]
    regTargeted= float(merge)/(combine+eps)
    print('new score  after pso= {}'.format(regTargeted))
    return regTargeted

def save_pso_contours(blueChannel_img,target,output,psooutput,output_img,oldimg,newimg,outimgBC_3,produceImageBifur):
    #orgimg blueChannel target outputimg outputBC outputContours outputPSOContours EllipseImg newInput
    pso_metric = np.zeros((256,256*9,3),np.uint8)
    emptyImage = np.zeros((256,256,3),np.uint8)
    emptyImage[:,:,:] = 255
    contours_l = [blueChannel_img,target,output_img,outimgBC_3,output,psooutput] 
    pso_metric[0:256,0:256,:] = oldimg
    for i in range(6):
        if i == 0:
            pso_metric[0:256,256*(i+1):256*(i+2),:] = blueChannel_img
            
        elif i ==2:
            pso_metric[0:256,256*(i+1):256*(i+2),:] = output_img
            
        elif i ==3:
            pso_metric[0:256,256*(i+1):256*(i+2),:] = outimgBC_3
            
        else:
            produceImage = emptyImage.copy()
            cv2.drawContours(produceImage, contours_l[i], -1, (0, 0, 0), -1)
            pso_metric[0:256,256*(i+1):256*(i+2),:] = produceImage
           
    pso_metric[0:256,256*7:256*8,:] = produceImageBifur
    pso_metric[0:256,256*8:256*9,:] = newimg
    return pso_metric

def createOutputContoursLoss(result,part):
    finalOutputContours = torch.zeros(result.size(0),1,400,1,2)
    for i in range(result.size(0)):
        out1 = result[i].squeeze().clone().detach() # [3, 256, 256]
        out1 = tensor2im(out1)
        outputcontoursLoss = get_img_contours(out1,part)
        outputcontoursLoss = creatContour(outputcontoursLoss, 400)
        outputcontoursLoss = torch.from_numpy(outputcontoursLoss)
        finalOutputContours[i] = outputcontoursLoss
    return finalOutputContours

def cal_IoU(GT_binary,mask_binary):
    #GT_image =  cv2.cvtColor(GT_binary, cv2.COLOR_BGR2RGB)
    GT_image = GT_binary.copy()
    GT_image[np.where(GT_image<255)] = 1
    GT_image[np.where(GT_image==255)] = 0
    GT_image_01 = GT_image#[:,:,0]
    #Bif_image 震荡之后的图片
    #Bif_image =  cv2.cvtColor(mask_binary, cv2.COLOR_BGR2RGB)
    Bif_image = mask_binary.copy()
    Bif_image[np.where(Bif_image<255)] = 1
    Bif_image[np.where(Bif_image==255)] = 0
    Bif_image_01 = Bif_image#[:,:,0]

    mix_image = GT_image_01 + Bif_image_01

    ##获得交/并的回归值
    merge  = np.where(mix_image==2)[0].shape[0]
    combine = np.where(mix_image>0)[0].shape[0]
    eps = 1e-7
    regTargeted= float(merge)/(combine+eps)

    regTargeted = np.array([regTargeted])
    regTargeted = torch.from_numpy(regTargeted)
    return GT_image,Bif_image,regTargeted

def cal_Dice(GT_binary,mask_binary):
    GT_image = GT_binary.copy()
    GT_image[np.where(GT_image<255)] = 1
    GT_image[np.where(GT_image==255)] = 0
    GT_image_01 = GT_image#[:,:,0]
    Bif_image = mask_binary.copy()
    Bif_image[np.where(Bif_image<255)] = 1
    Bif_image[np.where(Bif_image==255)] = 0
    Bif_image_01 = Bif_image#[:,:,0]

    mix_image = GT_image_01 + Bif_image_01

    ##获得交/并的回归值
    merge  = np.where(mix_image==2)[0].shape[0]
    combine = np.where(mix_image>0)[0].shape[0]
    eps = 1e-7
    regTargeted= 2*float(merge)/(combine+merge+eps)

    regTargeted = np.array([regTargeted])
    regTargeted = torch.from_numpy(regTargeted)
    return GT_image,Bif_image,regTargeted


def test(basemodel,model,testloader, test_save_path, epoch2, pso_num, n_particles, half=False,part='OD',epoch=None):
    print("PSO TEST")
    basemodel.eval()
    model.eval()
    with torch.no_grad():
        num = 0
        ### PSO参数初始化
        IND_DIM = 200+4 # 粒子维度
        for batch_idx,sample in enumerate(testloader):
            img = sample['image']
            mylabel = sample['mylabel'] 
            mylabel_rand = sample['mylabel_rand'] 
            imagename = sample['imagename']
            part = sample['part'][0]
            train = sample['train'] 
            mask = sample['mask'] 
            regTargeted = sample['regTargeted']
            data = img.cuda() 
            target = mylabel.cuda() 
            
            output,output_e,output_mid,output_d,l0,l1,l2,l3 = basemodel(data)

            for i, out in enumerate(output):
                num += 1
                out_list = []
                dice_v_list = []
                PSO_Score_list = []
                pos_20 = False
                outimgname_path = os.path.join(test_save_path, imagename[i].split(".")[0]) # 创建该图像输出路径
                if os.path.exists(outimgname_path):
                    #print('pass',outimgname_path)
                    continue
                if not os.path.exists(outimgname_path):
                    os.makedirs(outimgname_path)

                orgimg_tensor = data[i].squeeze().detach().cpu().numpy()
                orgimg = torch.squeeze(data[i]).detach().cpu()
                orgimg = tensor2im(orgimg)
                cv2.imwrite(os.path.join(outimgname_path,'img.jpg'),orgimg)
                target_t = target[i].squeeze().detach().cpu()
                cv2.imwrite(os.path.join(outimgname_path,'Display_GT_label.bmp'),np.abs(128*target_t.numpy()-255))

        
                hanetout = out.squeeze().detach().cpu()
                hanetout_arg = np.argmax(hanetout.numpy(), axis=0)
                display_hanetout = np.uint8(np.abs(128*hanetout_arg-255))
                cv2.imwrite(os.path.join(outimgname_path,"Display_basemodelout.bmp"),display_hanetout)
                
              
                contours_hanetout = get_contours_from_image(display_hanetout,part) 
                contours_GT = get_contours_from_image(np.uint8(np.abs(128*target_t-255)),part) 
                if len(contours_hanetout) == 1:
                    contours2 = creatContour(contours_hanetout, IND_DIM-4) ##构建统一边界点个数的contour
                else:
                    print("create model output_img contours error, break!")
                    break
                for e in range(epoch2):
                    start = time.time()
                    pos, pos_20, pop, fitness,gbestfitness = pso(pso_num, n_particles, IND_DIM, orgimg, contours2, model, half, pos_20,part)
                    end = time.time()
                    print('time=',end-start)
                    print("pso:[{}/{}]".format(e+1, epoch2))
                    #contours_after_pso = data_for_unet.basedGeometry2Contour(pos, contours_output) # 边界变化
                    contours_after_pso = basedGeometry2Contour(pos, contours_hanetout) # 边界变化
                    
                    emptyImage = orgimg.copy()
                    emptyImage[:,:,:] = 255

                    produceImgTarget = emptyImage.copy()
                    cv2.drawContours(produceImgTarget,contours_GT, -1, (0, 0, 0), -1) 

                    produceImgBeforePSO = emptyImage.copy()
                    cv2.drawContours(produceImgBeforePSO,contours_hanetout, -1, (0, 0, 0), -1) 


                    produceImgAfterPSO = emptyImage.copy()
                    cv2.drawContours(produceImgAfterPSO,contours_after_pso, -1, (0, 0, 0), -1) 

                    
                    _,_,regTargeted_before = cal_Dice(produceImgBeforePSO,produceImgTarget)
                    _,_,regTargeted_target = cal_Dice(produceImgAfterPSO,produceImgTarget)

                    if part == 'OD':
                        produceImgAfterPSO[np.where(produceImgAfterPSO==0)]= 128
                        #produceImgAfterPSO[np.where(produceImgAfterPSO==0)]= 255
                    if part =='OC':
                        produceImgAfterPSO[np.where(produceImgAfterPSO==0)]= 128
                        #produceImgAfterPSO[np.where(produceImgAfterPSO==0)]= 255

                    cv2.imwrite(os.path.join(outimgname_path, part+'_InputDice_'+str(regTargeted_before.data.item())[:5]+'_outputDice_'+str(regTargeted_target.data.item())[:5]+'_PSODice_'+str(gbestfitness)[:8]+'_'+"%d.jpg"%e),produceImgAfterPSO)
                    # if epoch is not None:
                    #     cv2.imwrite(os.path.join(outimgname_path, 'Epoch_'+str(epoch)+'_'+part+'_Score_'+str(score[i].cpu().data.item())[:5]+'_diceScore_'+str(dice_v[i].data.item())[:5]+'_PSOBestScore_'+str(gbestfitness)[:5]+'_FianlScore_'+str(regTargeted_final)[:5]+'_'+"%d.jpg"%e), final_matrix)
                
                    # else:
                    #    cv2.imwrite(os.path.join(outimgname_path, part+'_InputDice_'+str(score[i].cpu().data.item())[:5]+'_outputDice_'+str(dice_v[i].data.item())[:5]+'_PSOBestDice_'+str(gbestfitness)[:5]+'_FianlDice_'+str(regTargeted_final.data.item())[:5]+'_'+"%d.jpg"%e),produceImgAfterPSO)
                
                    print("图像保存在: ", outimgname_path,batch_idx)


class myReLU(nn.Module):
    def __init__(self):
        super(myReLU,self).__init__()
    def forward(self,x):
        x = torch.clamp(0.5*x,min=-0.5)
        return x 

#HANet

class MyModel(nn.Module):
    def __init__(self,finetune_e = False,finetune_ha = False,finetune_d = False):
        super(MyModel, self).__init__()
      
        self.finetune_e = finetune_e
        self.finetune_ha = finetune_ha
        self.finetune_d = finetune_d 
        
      
        
        self.model = getattr(models, 'HANet')(module='HA_module', num_classes=3,
                            encoder='resnet101',
                            output_stride=16,
                            sync_bn=False,
                            hop=2,
                            delta=0.5,
                           )
       
        self.chconv = nn.Sequential(nn.Conv2d(6,32,kernel_size=1,stride=1,bias=True),
                                    nn.Conv2d(32,3,kernel_size=1,stride=1,bias=True),
                                    ) 
        self.relu = myReLU()
        self.l0ScoreConv = nn.Sequential(*[nn.Conv2d(64,8,kernel_size=3,stride=1),
                                        nn.BatchNorm2d(8,affine=True),
                                        ])
        self.l0Score = nn.Sequential(*[ nn.Linear(8*126*126,1024),
                                        self.relu
                                        ])

        self.l1ScoreConv = nn.Sequential(*[nn.Conv2d(256,32,kernel_size=3,stride=1),
                                        nn.BatchNorm2d(32,affine=True),
                                        ])
        self.l1Score = nn.Sequential(*[ nn.Linear(32*62*62,1024),
                                        self.relu
                                        ])

        self.l2ScoreConv = nn.Sequential(*[nn.Conv2d(512,64,kernel_size=3,stride=1),
                                        nn.BatchNorm2d(64,affine=True),
                                        ])
        self.l2Score = nn.Sequential(*[ nn.Linear(64*30*30,1024),
                                        self.relu
                                        ])  

        self.l3ScoreConv = nn.Sequential(*[nn.Conv2d(1024,128,kernel_size=3,stride=1),
                                        nn.BatchNorm2d(128,affine=True),
                                        ])
        self.l3Score = nn.Sequential(*[ nn.Linear(128*14*14,1024),
                                        self.relu
                                        ])
        
        self.conv2linear = nn.Sequential(*[nn.Conv2d(256,32,kernel_size=3,stride=1),
                                nn.BatchNorm2d(32,affine=True),
                                ])
       
        self.score = nn.Sequential(*[
                                    nn.Linear(32*14*14,1024),
                                    self.relu,
                                    ])
        self.score_final = nn.Sequential(*[
                                    nn.Linear(1024*5,512),
                                    self.relu,
                                    nn.Linear(512,256),
                                    self.relu,
                                    nn.Linear(256,1),
                                    self.relu
                                    ])
 
    def forward(self,x):
        x = self.chconv(x)
        x,output_e,output_mid,output_d,l0,l1,l2,l3 = self.model(x)
        score_l0_feat = self.l0ScoreConv(l0)
        score_l0 = self.l0Score(score_l0_feat.view(score_l0_feat.size(0),-1))

        score_l1_feat = self.l1ScoreConv(l1)
        score_l1 = self.l1Score(score_l1_feat.view(score_l1_feat.size(0),-1))

        score_l2_feat = self.l2ScoreConv(l2)
        score_l2 = self.l2Score(score_l2_feat.view(score_l2_feat.size(0),-1))

        score_l3_feat = self.l3ScoreConv(l3)
        score_l3 = self.l3Score(score_l3_feat.view(score_l3_feat.size(0),-1))

        score_mid_feat = self.conv2linear(output_mid)
      
        score_mid = self.score(score_mid_feat.view(score_mid_feat.size(0),-1))
        wait_to_cat = [score_l0,score_l1,score_l2,score_l3,score_mid]
        score_all = torch.cat(wait_to_cat,dim=1)
        final_score = self.score_final(score_all)

        return x,final_score






def main():
    #os.environ["CUDA_VISIBLE_DEVICES"] = '1' 
    config = configparser.ConfigParser()
    config.read("config.ini", encoding='utf-8') 
    model_load_dir = config.get('DEFAULT','model_load_dir')
    if model_load_dir == 'False':
        model_load_dir = False
    is_test = config.getboolean('DEFAULT','is_test')

    part = config.get('DEFAULT','part')
    half = config.getboolean('DEFAULT','half') 
    path_test = config.get('DEFAULT','path_test')
    test_save_path = config.get('DEFAULT','test_save_path')
    ### PSO参数初始化
    psomodel_epoch_num = config.getint('DEFAULT','psomodel_epoch_num')
    pso_epoch_num = config.getint('DEFAULT','pso_epoch_num')
    n_particles = config.getint('DEFAULT','n_particles')

 
    
   

    testset = REFUGE.MyTestDataset(path_test,Train = False,part=part)

    testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                              shuffle=True,num_workers=2)

    cudnn.benchmark = True if torch.cuda.is_available() else False
    
  
    basemodel = getattr(models, 'HANet')(module='HA_module', num_classes=3,
                            encoder='resnet101',
                            output_stride=16,
                            sync_bn=False,
                            hop=2,
                            delta=0.5,
                            )
  
    
    basemodel_ckpt = torch.load('./ckpt_e100_loss_0.1714.pth.tar')
    #basemodel_ckpt = torch.load('/datastore/wait_to_review/sunyz/code/pso_seg/radial_grad/segmodel/HANet/OC/v_original/ckpt_e100_loss_0.0501.pth.tar')
    basemodel.load_state_dict(basemodel_ckpt["state_dict"])
    
   
    model = MyModel()
    state_dict = torch.load(model_load_dir)
    model.load_state_dict(state_dict['state_dict'])
   
    model = model.cuda() 
    basemodel = basemodel.cuda()
   

   
    if is_test: # 测试
        print("测试集为：", path_test)
        print("开始测试：")
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)
        test(basemodel,model, testloader, test_save_path, psomodel_epoch_num, pso_epoch_num, n_particles, half,part)
    
if __name__ == "__main__":
    main()
