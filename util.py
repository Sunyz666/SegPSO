import numpy as np
import torch
from torchvision import transforms
import math
from PIL import Image
import cv2
import os

import dataPSOTest

import matplotlib.pyplot as plt

def tensor2im(input_image, imtype=np.uint8):
   
    mean = [0.485, 0.456, 0.406] #自己设置的
    std = [0.229, 0.224, 0.225]  #自己设置的
    
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        image_numpy = image_numpy[:,:,::-1]  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def read_json(file_path):
    res = []
    with open(file_path, 'r') as o:
        data = o.readlines()
    for i in data:
        j = json.loads(i.strip())
        res.append(j)
    return res
    
# 用眼盘边缘替换原图的blue通道
def contour2LightMaskImage(reContours, orgimg):
    emptyImage = orgimg.copy()
    emptyImage[:,:,:] = 255
    produceImage = emptyImage.copy()
    reContours = reContours.astype(np.int)
    cv2.drawContours(produceImage, reContours, -1, (0, 0, 0), -1)

    Bif_image = cv2.cvtColor(produceImage, cv2.COLOR_BGR2RGB)
    Bif_image[np.where(Bif_image < 255)] = 1
    Bif_image[np.where(Bif_image == 255)] = 0
    Bif_image_01 = Bif_image#[:, :, 0]

    blueChannel= Bif_image_01 * 255 
    #cv2.imwrite('asdasdas.jpg',orgimg)
    return orgimg,blueChannel

# 用眼盘边缘替换原图的blue通道
def contour2LightMaskImage2(reContours, orgimg):
    # emptyImage = cv2.imread('empty.jpg')
    emptyImage = orgimg.copy()
    emptyImage[:,:,:] = 255
    produceImage = emptyImage.copy()

    reContours = reContours.astype(np.int)
    cv2.drawContours(produceImage, reContours, -1, (0, 0, 0), -1)

    Bif_image = cv2.cvtColor(produceImage, cv2.COLOR_BGR2RGB)
    Bif_image[np.where(Bif_image < 255)] = 1
    Bif_image[np.where(Bif_image == 255)] = 0
    Bif_image_01 = Bif_image[:, :, 0]
    orgimg[:, :, 0] = Bif_image_01 * 255
    return orgimg, Bif_image_01

def creatContour(contour, N):
    newContour = np.zeros((1,N,1,2))
    _,csize,_,_ = contour.shape 
    strip = (csize-1)/(N-1)
    for i in range(N):
        newContour[0,i,0,:] = contour[0,int(i*strip),0,:]
    newContour[0,-1,0,:] = contour[0,-1,0,:]

    return newContour.astype(int)

def Nrotate(angle,valuex,valuey,pointx,pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
    nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
    return nRotatex, nRotatey

WH = 10
BIAS_WH = 10


def basedGeometry2Contour(keyPoints, mycontours):

    # print("传入的关键点", keyPoints.shape, len(keyPoints))
    _, allPointNum, _, _ = mycontours.shape
    # print(mycontours.shape)
    refContours= mycontours.reshape((allPointNum,2))
    meanCenter = refContours.mean(axis=0)
    # print("centers:",meanCenter.shape, refContours.shape)

    ##将参考轨迹的中心移动到此中心位置，形成坐标系的转换。
    refContours = refContours - meanCenter

    # print(refContours)


    moveBias = keyPoints[0], keyPoints[1]  # x,y bias
    moveBias = np.array(moveBias)
    # moveBias = moveBias - 0.5  ## front or back move
    moveBias = moveBias * BIAS_WH   ## WH image size

    ##scale parameter
    scale = keyPoints[2] + 1

    ##rotate angle, need exchange
    rotate = keyPoints[3]

    ##

    ##大小跟参考contour个数一致

    insertPoints = keyPoints[4:]
    insertPoints = np.array(insertPoints)
    insertPoints = insertPoints  * WH  ##distance

    ## get numbers of points and key points
    # print(insertPoints.shape)
    insNum = len(insertPoints)

    strip = float(allPointNum)/insNum
    # print("insNum", insNum)
    #assert()
    lookHT = 30

    retContour = np.zeros((1,insNum,1,2))
    #cache = np.zeros((insNum+1,2))
    #cache[0,0] = np.random.uniform(-5.,5.)
    #cache[0,1] = np.random.uniform(-5.,5.)
    for i in range(insNum):
        indx = int(i * strip)
        x0  = refContours[indx,0]
        y0  = refContours[indx,1]
        d0 = math.sqrt(x0 * x0 + y0 * y0)

        biasDis = 0
        for j in range(lookHT):
            biasDis = biasDis + (insertPoints[i-j] + insertPoints[(i+j)%insNum])
        biasDis = insertPoints[i] + biasDis
        biasDis = biasDis/(lookHT * 2 + 1)

        retContour[0, i, 0, 0] = (biasDis + d0) * (x0/d0) #+ cache[i,0]
        retContour[0, i, 0, 1] = (biasDis + d0) * (y0/d0) #+ cache[i,1]
        
        #cache[i+1,0] = retContour[0, i, 0, 0]/100
        #cache[i+1,1] = retContour[0, i, 0, 1]/100
    ##scale
    retContour = scale * retContour

    # print(retContour.shape)
    ## rotate
    retContour[0, :, 0, 0] , retContour[0,:,0, 1] = Nrotate(rotate, retContour[0,:,0,0], retContour[0,:,0,1] , 0, 0)
    ## move back
    retContour[0, :, 0, :] = retContour[0, :, 0, :] + meanCenter

    retContour[0,:,0,:] = retContour[0,:,0,:] + moveBias
    retContour = (retContour.astype(np.int))

    return retContour


def get_center_pos(blueChannelImage):
    blueChannelImage =  cv2.cvtColor(blueChannelImage, cv2.COLOR_BGR2GRAY)
    blueChannelContours, blueChannelHierarchy = cv2.findContours(blueChannelImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    assert len(blueChannelContours) >= 1
    contours2 = []
    for n in range(len(blueChannelContours)):
        cnt = blueChannelContours[n]
        area = cv2.contourArea(cnt)
        contours2.append([cnt,area])
    contours2 = sorted(contours2, key=lambda x:x[1])
    blueChannelContours = np.array([contours2[0][0]])
    blueChannelarea = contours2[0][1]
    #blueChannelContours = np.array([blueChannelContours[0]])

    _, allPointNum, _, _ = blueChannelContours.shape
    refContours= blueChannelContours.reshape((allPointNum,2))
    meanCenter = refContours.mean(axis=0)
    return np.round(meanCenter),blueChannelarea

def get_dis(contours,GT):
    allPointNum, _, _ = contours.shape
    refContours= contours.reshape((allPointNum,2))
    meanCenter = refContours.mean(axis=0)
    return math.sqrt((meanCenter[0]-GT[0])**2 + (meanCenter[1]-GT[1])**2)


def cal_threshold(gray,part):
    if part == 'OD':
        gray[np.where(gray==0)]=200
        ret, threshold = cv2.threshold(gray,200, 200, cv2.THRESH_BINARY)
    if part == 'OC':
        #gray[np.where(gray >=120 )] = 255
        ret, threshold = cv2.threshold(gray, 100, 255,cv2.THRESH_BINARY)  
        gray[np.where(gray==0)]=200
        ret, threshold = cv2.threshold(gray,200, 200, cv2.THRESH_BINARY)
        
    return ret, threshold

def get_contours_from_image(GT,part):
    gray = GT.copy() 
    ret, threshold = cal_threshold(gray,part) #get res & threshold 
    #cv2.imwrite('eeqweqwe.bmp',threshold)
    contours2 = []
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for n in range(len(contours)):
        cnt = contours[n]
        area = cv2.contourArea(cnt)
        #print(area)
        if area >=400 and part=='OC':
            contours2.append([cnt,area])
        if area >=800 and part=='OD':
            contours2.append([cnt,area])

    assert len(contours2)>0
    contours2 = sorted(contours2, key=lambda x:x[1])
    ct = np.array([contours2[0][0]])
    ct2 = ct[:,:,:,:]
    contours = ct2
    return contours

def get_img_contours(labelimg,part,blueChannel=None):
    ##计算回归值
    # blueChannel = [-1,1]
    if blueChannel is not None:
        try:
            meanCenter,blueChannelarea = get_center_pos(blueChannel)
            print('get')
        except:
            #cv2.imwrite(os.path.join('bad blueChannel'),blueChannel)
            return get_img_contours(labelimg,part)
        gray =  cv2.cvtColor(labelimg, cv2.COLOR_BGR2GRAY)
        if part == 'OD':
            gray[np.where(gray==0)]= 200
            ret, threshold = cv2.threshold(gray, 200, 200,0)
            ##已获得的边界区域
            contours2 = []
            ###cv2 4.0 del binary,
            contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            assert len(contours) >= 1
            flag = 0
            for n in range(len(contours)):
                cnt = contours[n]
                area = cv2.contourArea(cnt)
                #print('blue area',area)
                if 100<=area<=200000 :
                    Dis = get_dis(cnt,meanCenter)
                    contours2.append([cnt,Dis])
            if len(contours2) == 0:
                flag = 1
                for n in range(len(contours)):
                    cnt = contours[n]
                    area = cv2.contourArea(cnt)
                    #print('blue len=0 area',area)
                    Dis = get_dis(cnt,meanCenter)
                    contours2.append([cnt,Dis])

            contours2 = sorted(contours2, key=lambda x:x[1])
            contours = np.array([contours2[0][0]]) if flag == 0 else np.array([contours2[-1][0]])

            emptyImage = labelimg.copy()
            emptyImage[:, :, :] = 255
            produceImage = emptyImage.copy()

            # try:
            #     newcontours = contours[0].reshape(-1,2)
            #     newcontours = cv2.fitEllipse(newcontours)
            #     cv2.ellipse(produceImage, newcontours, (0,0,0), -1)
            #     produceImagegray =  cv2.cvtColor(produceImage, cv2.COLOR_BGR2GRAY)
            #     produceImagegray[np.where(produceImagegray==0)]= 200
            #     ret, produceImagethreshold = cv2.threshold(produceImagegray, 200, 200,0)
            #     contours, hierarchy = cv2.findContours(produceImagethreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #     assert len(contours) >= 1
            #     flag = 0
            #     for n in range(len(contours)):
            #         cnt = contours[n]
            #         area = cv2.contourArea(cnt)
            #         if 100<=area<=200000 :
            #             Dis = get_dis(cnt,meanCenter)
            #             contours2.append([cnt,Dis])
                 
            #     if len(contours2) == 0:
            #         flag = 1
            #         for n in range(len(contours)):
            #             cnt = contours[n]
            #             area = cv2.contourArea(cnt)
            #             Dis = get_dis(cnt,meanCenter)
            #             contours2.append([cnt,Dis])

            #     contours2 = sorted(contours2, key=lambda x:x[1])
            #     contours = np.array([contours2[0][0]]) if flag == 0 else np.array([contours2[-1][0]])

            # except:
            #     contours = contours

          
        elif part == 'OC':
            gray[np.where(gray >= 130)] = 255  # OC
            ret, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)  # [0,128,200]
            ##已获得的边界区域
            contours2 = []
            ###cv2 4.0 del binary,
            contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for n in range(len(contours)):
                # 筛选面积较大的连通区，阈值为20000
                cnt = contours[n]
                area = cv2.contourArea(cnt)

                if area>=50 and area<=200000:  #if area>=200 and area<=200000:
                    contours2.append(cnt)
            contours = sorted(contours2, key=lambda x: len(x))
            assert len(contours) > 0
            ct = np.array([contours[0]])
            ct2 = ct[:, :, :, :]
            contours = ct2
    else:
        gray =  cv2.cvtColor(labelimg, cv2.COLOR_BGR2GRAY)
        if part == 'OD':
            gray[np.where(gray==0)]= 200
            ret, threshold = cv2.threshold(gray, 200, 200,0)
            ##已获得的边界区域
            contours2 = []
            ###cv2 4.0 del binary,
            contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            assert len(contours) >= 1
            flag = 0
            for n in range(len(contours)):
                cnt = contours[n]
                area = cv2.contourArea(cnt)
                #print('area',area)
                if 1000 <= area <= 100000: #if 400 <= area <= 200000:
                    contours2.append([cnt,area])
            
            if len(contours2) == 0:
                flag = 1
                for n in range(len(contours)):
                    cnt = contours[n]
                    area = cv2.contourArea(cnt)
                    contours2.append([cnt,area])

            contours2 = sorted(contours2, key=lambda x:x[1])
            contours = np.array([contours2[0][0]]) if flag == 0 else np.array([contours2[-1][0]])

            emptyImage = labelimg.copy()
            emptyImage[:, :, :] = 255

            # try:
            #     newcontours = contours[0].reshape(-1,2)
            #     newcontours = cv2.fitEllipse(newcontours)
            #     cv2.ellipse(produceImage, newcontours, (0,0,0), -1)
            #     produceImagegray =  cv2.cvtColor(produceImage, cv2.COLOR_BGR2GRAY)
            #     produceImagegray[np.where(produceImagegray==0)]= 200
            #     ret, produceImagethreshold = cv2.threshold(produceImagegray, 200, 200,0)
            #     contours, hierarchy = cv2.findContours(produceImagethreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #     assert len(contours) >= 1
            #     flag = 0
            #     for n in range(len(contours)):
            #         cnt = contours[n]
            #         area = cv2.contourArea(cnt)
            #         if 400 <= area <= 200000:
            #             contours2.append([cnt,area])
                
            #     if len(contours2) == 0:
            #         flag = 1
            #         for n in range(len(contours)):
            #             cnt = contours[n]
            #             area = cv2.contourArea(cnt)
            #             contours2.append([cnt,area])

            #     contours2 = sorted(contours2, key=lambda x:x[1])
            #     contours = np.array([contours2[0][0]]) if flag == 0 else np.array([contours2[-1][0]])

            # except:
            #     contours = contours
                
            
        elif part == 'OC':
            gray[np.where(gray >= 130)] = 255  # OC
            ret, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)  # [0,128,200]
            # gray[np.where(gray==0)]= 200
            # ret, threshold = cv2.threshold(gray, 200, 200,0)
            ##已获得的边界区域
            contours2 = []
            ###cv2 4.0 del binary,
            contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for n in range(len(contours)):
                # 筛选面积较大的连通区，阈值为20000
                cnt = contours[n]
                area = cv2.contourArea(cnt)
                #print('OC area',area)
                if area>=200 and area<=20000:
                    contours2.append(cnt)
            contours = sorted(contours2, key=lambda x: len(x))
            assert len(contours) > 0
            ct = np.array([contours[0]])
            ct2 = ct[:, :, :, :]
            contours = ct2
    return np.array(contours)




def GradSPos(mycontours,RGBpath,centerfile='/datastore/wait_to_review/sunyz/code/pso_seg/radial_grad/REFUGE_cut_train_OD.json'):
    def  getranddis(x,y,res_all_dic):
        angle = round(math.atan2(x,y),2)
        try:
            choice_l = res_all_dic[angle]
        except:
            print('choice list error ')
            choice_l = [[0,0]]

        idx = np.random.choice(len(choice_l))
        return (x+choice_l[idx][0])/2, (y+choice_l[idx][1])/2
        
    RGB = cv2.imread(RGBpath)
    image_b, image_g, image_r = cv2.split(RGB)
    data = read_json(centerfile)
    print(RGBpath)
    RGBname = RGBpath.split('/')[-1].rstrip('bmp')+'jpg'
    try:
        for item in data:
            if item["filename"] == RGBname:
                center = item["center"]
                break
    except:
        print('not find centerfile')

    emptyImage = RGB.copy()
    emptyImage[:,:] = 255
    produceImage = emptyImage.copy()
    cv2.drawContours(produceImage, mycontours, -1, (0, 0, 0), 1)
    row,col = np.where(produceImage==0)
    circle = []
    for x,y in zip(list(row),list(col)):
        if x in [0,255] or y in [0,255]:continue
        circle.append([x,y])
    
    #Gra_pso = Gradient(image_g,center=(center[0],center[1]),dotList = circle,align = False)
    Gra_all = Gradient(image_g,center=(center[0],center[1]),dotList = None,align = False)
    img_all = Gra_all.sobel()
    #img_pso = Gra_pso.sobel()

    img_all_list  = []
    img_pso_list = circle

    row_all,col_all = np.where(img_all==0)
    for x,y in zip(list(row_all),list(col_all)):
        if x in [0,255] or y in [0,255]:continue
        img_all_list.append([x,y])
    img_all_list.sort(key=lambda x:atan2(x[0],x[1]),reverse=True)
    img_pso_list.sort(key=lambda x:atan2(x[0],x[1]),reverse=True)
    res_all,res_pso = [],[]

    res_all_dic,res_pso_dic = collections.OrderedDict(), collections.OrderedDict()
    for item in img_all_list:
        if not res_all_dic.__contains__(round(atan2(item[0],item[1]),2)):
            res_all_dic[round(atan2(item[0],item[1]),2)] = [item]
        else:
            res_all_dic[round(atan2(item[0],item[1]),2)] .append(item)
    for item in img_pso_list:
        if not res_pso_dic.__contains__(round(atan2(item[0],item[1]),2)):
            res_pso_dic[round(atan2(item[0],item[1]),2)] = [item]
        else:  
            res_pso_dic[round(atan2(item[0],item[1]),2)].append(item)

    #res_pso_dic res_all_dic : [[x,y],...,[x,y]]的位置list，pso是mycontours得到的，all是原图的所有点

    _, allPointNum, _, _ = mycontours.shape
    refContours= mycontours.reshape((allPointNum,2))

    insertPoints = refContours
    insertPoints = np.array(insertPoints)
    #insertPoints = insertPoints  * WH  ##distance
    ## get numbers of points and key points
    insNum = allPointNum
    strip = float(allPointNum)/insNum
 
    #lookHT = 30

    retContour = np.zeros((1,insNum,1,2))
   

    for i in range(insNum):
        indx = int(i * strip)
        x0  = refContours[indx,0]
        y0  = refContours[indx,1]
        d0 = math.sqrt(x0 * x0 + y0 * y0)
        biasDis = getranddis(x0,y0,res_pso_dic,res_all_dic)
        biasDisx0 = x0+ biasDis[0]
        biasDisy0 = y0+ biasDis[1]

        # retContour[0, i, 0, 0] = (biasDisx0 + d0) * (x0/d0) 
        # retContour[0, i, 0, 1] = (biasDisy0 + d0) * (y0/d0) 
        retContour[0, i, 0, 0] = biasDisx0
        retContour[0, i, 0, 1] = biasDisy0
        
    ##scale
    #retContour = scale * retContour

    # print(retContour.shape)
    ## rotate
    #retContour[0, :, 0, 0] , retContour[0,:,0, 1] = Nrotate(rotate, retContour[0,:,0,0], retContour[0,:,0,1] , 0, 0)

    # print(meanCenter)
    ## move back
    #retContour[0, :, 0, :] = retContour[0, :, 0, :] + meanCenter

    #retContour[0,:,0,:] = retContour[0,:,0,:] + moveBias
    retContour = (retContour.astype(np.int))

    # print(refContours.shape)
    return retContour





def dl_score_with_lightmask(x, a, b, c, half=False):
    n_particles = len(x)
    model = c
    model.eval()
    x = x.reshape((n_particles, -1))
    w,h,c = a.shape
    reimage = np.zeros((n_particles, w, h ,3))
    blueChannel = np.zeros((n_particles, w, h,3))

    for i in range(n_particles):
        reContours = basedGeometry2Contour(x[i], b) # 根据粒子随机对原始标记位置改变
       
        reimage[i],blueChannel[i] = contour2LightMaskImage(reContours, a) # 将标记加入到原图上


    
    testset = dataPSOTest.MyDataset(reimage, blueChannel)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=1)

    predicts = []
    for data, mask,target in testloader:
        with torch.no_grad():
            if half:
                data = data.half()
            data, mask, target = data.cuda(), mask.cuda() , target.cuda()
            new6Cdata = torch.cat((data,mask),1)
            output, dice_v = model(new6Cdata)
            predicts.append(dice_v.cpu().detach().numpy().flatten().tolist())
    
    predicts = np.array(predicts)
    predicts = predicts.flatten()
   
    return predicts



def dl_score_with_lightmask_test(x, a, b, c, half, path, criterion): 
    #print(x.shape,"输入个体的情况")
    n_particles = len(x)
    model = c
    model.eval()
    x = x.reshape((n_particles, -1))
    w,h,c = a.shape
    #print(x.shape,"输入个体的情况")

    mean1=(0.5, 0.5, 0.5)
    std1=(0.5, 0.5, 0.5)
    reimage = np.zeros((n_particles, w, h ,3))
    target_list = []
    for i in range(n_particles):
        reContours = basedGeometry2Contour(x[i], b) # 根据粒子随机对原始标记位置改变
        reimage[i], Bif_image_01 = contour2LightMaskImage2(reContours, a) # 将标记加入到原图上
        _, GT_image_01 = contour2LightMaskImage2(b, a) # GT图像
        mix_image = GT_image_01 + Bif_image_01
        ##获得交/并的回归值
        eps = 1e-4
        merge  = np.where(mix_image==2)[0].shape[0]
        combine = np.where(mix_image>0)[0].shape[0]
        regTargeted= float(merge)/(combine+eps)
        target_list.append(regTargeted)

    testset = dataPSOTest.MyTestDataset(reimage, target_list, transform=transforms.Compose([
        transforms.Resize((512, 512), Image.BILINEAR),
        # transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean1, std=std1)
    ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=1)

    predicts = []
    for i, (data, target) in enumerate(testloader):
        with torch.no_grad():
            if half:
                data = data.half()
            data, target = data.cuda(), target.cuda()
            output, dice_v = model(data)
            predicts.append(dice_v.cpu().detach().numpy().flatten().tolist())
            l1 = criterion(output.float(), data.float()).data.item()
            l2 = criterion(dice_v.float(), target.float()).data.item()
            loss = l1 + l2

            output = torch.squeeze(output, 0) # [3, 512, 512]
            output = tensor2im(output)  ###### 调整
            #out_path = os.path.join(path, "_"+str(i))#%num
            data = torch.squeeze(data, 0)
            data = tensor2im(data)
            target = float(target.cpu().detach().item())
            p = float(dice_v.cpu().detach().item())
            cv2.imwrite(os.path.join(path, "in_{}_{}.jpg".format(i+1, target)), data)
            cv2.imwrite(os.path.join(path, "out_{}_{}_{}.jpg".format(i+1, p, loss)), output)

            contours = get_img_contours(output) # 获取边界
            out_img = contour2LightMaskImage(contours, a) # 转换图像蓝色通道
            cv2.imwrite(os.path.join(path, "out2_{}.jpg".format(i+1)), out_img)
    
    predicts = np.array(predicts)
    predicts = predicts.flatten()
    #print(predicts)
    #predicts = predicts[0] * (-1)
    #print("predicts:", predicts)

    return predicts



class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
    
    img = './SegModel_results/HA_REFUGECUT_ODmask_ohemloss_lr_adaparam/39/0_472_526_997_V0128/Display_output.bmp'
    img = cv2.imread(img,0)
    contours = get_contours_from_image(img,'OD')