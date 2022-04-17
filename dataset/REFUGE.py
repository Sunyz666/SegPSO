import os
import torch
import numpy as np
from torchvision import transforms,utils
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import cv2
import math
#from .import util 
import matplotlib.pyplot as plt
import dataset.mytransform as transfor
import dataset.mytransformRand as transforRand
#import dataset.distort

def default_loader(path):
    return Image.open(path).convert('RGB')

def Nrotate(angle,valuex,valuey,pointx,pointy):
 valuex = np.array(valuex)
 valuey = np.array(valuey)
 nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
 nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
 return nRotatex, nRotatey

WH = 10
BIAS_WH = 10
### 以法向量为基准，形成的图像边界的变化，形成图像演化
def basedGeometry2Contour(keyPoints, mycontours):
    
    #print("传入的关键点", keyPoints.shape, len(keyPoints))
    _, allPointNum, _, _ = mycontours.shape
    # print(mycontours.shape)
    refContours= mycontours.reshape((allPointNum,2))
    meanCenter = refContours.mean(axis=0)
    # print("centers:",meanCenter.shape, refContours.shape)

    ##将参考轨迹的中心移动到此中心位置，形成坐标系的转换。
    refContours = refContours - meanCenter



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

    cache = np.zeros((insNum+1,2))
    cache[0,0] = np.random.uniform(-5.,5.)
    cache[0,1] = np.random.uniform(-5.,5.)
    for i in range(insNum):
        indx = int(i * strip)
        x0  = refContours[indx,0]
        y0  = refContours[indx,1]
        d0 = math.sqrt(x0 * x0 + y0 * y0)

        biasDis = 0
        for j in range(lookHT):
            biasDis += (insertPoints[i-j] + insertPoints[(i+j)%insNum])
        biasDis = insertPoints[i] + biasDis
        biasDis = biasDis/(lookHT * 2 + 1)

       
        retContour[0, i, 0, 0] = (biasDis + d0) * (x0/d0) #+ cache[i,0]
        retContour[0, i, 0, 1] = (biasDis + d0) * (y0/d0) #+ cache[i,1]
        
        cache[i+1,0] = retContour[0, i, 0, 0]/100
        cache[i+1,1] = retContour[0, i, 0, 1]/100
        #print(cache[i+1])
    ##scale
    retContour = scale * retContour

    # print(retContour.shape)
    ## rotate
    retContour[0, :, 0, 0] , retContour[0,:,0, 1] = Nrotate(rotate, retContour[0,:,0,0], retContour[0,:,0,1] , 0, 0)

    # print(meanCenter)
    ## move back
    retContour[0, :, 0, :] = retContour[0, :, 0, :] + meanCenter

    retContour[0,:,0,:] = retContour[0,:,0,:] + moveBias
    retContour = (retContour.astype(np.int))

    # print(refContours.shape)
    return retContour


def read_json(file_path):
    res = []
    with open(file_path, 'r') as o:
        data = o.readlines()
    for i in data:
        j = json.loads(i.strip())
        res.append(j)
    return res


class Gradient(object):

    # 初始化参数
    # img：输入图片；threshold：幅值；radium：半径；theta：角度； num_p：环上点的个数
    def __init__(self, img, center=(100,128),img_ves=None,dotList = None,align=None):
        """
        初始化参数
        :param img: ROI图
        :param img_ves: vessel图，用于去除干扰
        :param param: radius：选取点的间隔；    num_p：环上点的个数，即子群的个数
        """
        self._img = img
        self._img_ves = img_ves
        self._radius = 3
        self._num_p = 200
        self.mag = np.zeros(self._img.shape)
        self.gra_max = 0
        self.gra_min = 15
        self.centerw = center[0]
        self.centerh = center[1]
        self.dot = dotList
        self.align = align
        # params = parameters.load_para()
        # self.mag = np.zeros(((self._img, 0), (self._img, 0)), dtype=int)
    def sobel(self):
        g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        g_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        if self.dot is None:
            rows = np.size(self._img, 0)
            columns = np.size(self._img, 1)
            for i in range(self.gra_min, int(rows / 2 / self._radius) - self.gra_max):  # radius
                for j in range(0, int(self._num_p)):  # theta
                    x = int(round(i * self._radius * math.cos(2 * j * math.pi / self._num_p) + self.centerw))
                    y = int(round(i * self._radius * math.sin(2 * j * math.pi / self._num_p) + self.centerh))
                
                    #print(x,y)
                    if 4 < x + 2 < int(rows) - 2 and 4 < y + 2 < int(rows) - 2:
                        v = sum(sum(np.multiply(g_x, self._img[x - 1:x + 2, y - 1:y + 2])))  # vertical
                        h = sum(sum(np.multiply(g_y, self._img[x - 1:x + 2, y - 1:y + 2])))  # horizon
                        self.mag[x, y] = round(math.sqrt(math.pow(v, 2)) + math.sqrt(math.pow(h, 2)))
                    else:
                        break
                    # if (self._img_ves[x, y] != [0, 0, 0]).all():
                    #     self.mag[x, y] = 0
        elif self.dot is not None:
            rows = np.size(self._img, 0)
            columns = np.size(self._img, 1)
            if self.align:
                for i in range(self.gra_min, int(rows / 2 / self._radius) - self.gra_max):  # radius
                    for j in range(0, int(self._num_p)):  # theta
                        x = int(round(i * self._radius * math.cos(2 * j * math.pi / self._num_p) + self.centerw))
                        y = int(round(i * self._radius * math.sin(2 * j * math.pi / self._num_p) + self.centerh))
                        if [x,y] in circle:
                            if 4 < x + 2 < int(rows) - 2 and 4 < y + 2 < int(rows) - 2:
                                v = sum(sum(np.multiply(g_x, self._img[x - 1:x + 2, y - 1:y + 2])))  # vertical
                                h = sum(sum(np.multiply(g_y, self._img[x - 1:x + 2, y - 1:y + 2])))  # horizon
                                self.mag[x, y] = round(math.sqrt(math.pow(v, 2)) + math.sqrt(math.pow(h, 2)))
                            else:
                                break
                        else:
                            continue
            else:
                for x,y in circle:
                    if 4 < x + 2 < int(rows) - 2 and 4 < y + 2 < int(rows) - 2:
                        v = sum(sum(np.multiply(g_x, self._img[x - 1:x + 2, y - 1:y + 2])))  # vertical
                        h = sum(sum(np.multiply(g_y, self._img[x - 1:x + 2, y - 1:y + 2])))  # horizon
                        self.mag[x, y] = round(math.sqrt(math.pow(v, 2)) + math.sqrt(math.pow(h, 2)))
                    else:
                        break
        return self.mag

    # 归一化处理  梯度值>0;范围：(0,255)
    def img_norm(self):
        self.mag *= (self.mag > 0)
        self.mag = (self.mag - self.mag.min()) / (self.mag.max() - self.mag.min())
        self.mag = self.mag * 25
        return np.uint8(self.mag)



def Grad2Contour(keyPoints,mycontours,RGBpath,centerfile='/datastore/wait_to_review/sunyz/code/pso_seg/radial_grad/REFUGE_cut_train_OD.json'):
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
    #meanCenter = refContours.mean(axis=0)
   
    ##将参考轨迹的中心移动到此中心位置，形成坐标系的转换。
    #refContours = refContours - meanCenter

    #moveBias = keyPoints[0], keyPoints[1]  # x,y bias
    #moveBias = np.array(moveBias)
    # moveBias = moveBias - 0.5  ## front or back move
    #moveBias = moveBias * BIAS_WH   ## WH image size

    ##scale parameter
    #scale = keyPoints[2] + 1

    ##rotate angle, need exchange
    #rotate = keyPoints[3]

    #insertPoints = keyPoints[4:]
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


class MyDataset(Dataset):
    def __init__(self, root, Train, loader=default_loader,part='OD'):
        imgs = []
        labeldir  = os.path.join(root, "seg")
        for labelroot,dirs,files in os.walk(labeldir):
            for file in files:
                imgpath = (os.path.join(labelroot,file))
                imgs.append(imgpath)
        self.imgs = imgs
        self.root = root
        self.loader = loader
        self.part = part
        self.Train = Train
        self.transform_sample= self.transform_train if self.Train else self.transform_val
        print('Train= {}, part = {}'.format(self.Train,self.part))
       
    def __getitem__(self, index):
        ##图片放在一个目录中,利用bmp和jpg的后缀,区分GT 和 原图
        labelfn = self.imgs[index]
        imagename = os.path.basename(labelfn)
        imgdir  = os.path.join(self.root, "image")
        orgfn  = os.path.join(imgdir, imagename.rstrip("bmp")+"jpg")
       
        ##原图和标记图
        #GT = cv2.imread(labelfn,cv2.IMREAD_GRAYSCALE)
        GT =Image.open(labelfn)
        GT = np.array(GT)
        #orgimg = cv2.imread(orgfn)
        orgimg = Image.open(orgfn).convert('RGB')
        orgimg = np.array(orgimg)
        
        contours = self.get_contours_from_image(GT,self.part) #得到GT的contours

        mylabel = GT.copy() # 格式化OD和OC之后的label，RGB
        emptyImage = GT.copy()
        emptyImage[:,:] = 255
        produceImage = emptyImage.copy()
        produceImageBifur = emptyImage.copy()
        cv2.drawContours(produceImage, contours, -1, (0, 0, 0), -1)
        
        if np.random.rand()<0.1:
            bifur = -0.3 + 0.3 * np.random.rand(200+4,) #-1 + 2 * np.random.rand(50+4,) -0.5 0.5
        elif 0.1<=np.random.rand()<0.3:
            bifur = -0.22 + 0.22 * np.random.rand(200+4,) 
        elif 0.3<=np.random.rand()<0.6:
            bifur = -0.15 + 0.15 * np.random.rand(200+4,) 
        elif 0.6<=np.random.rand()<0.8:
            bifur = -0.10 + 0.10 * np.random.rand(200+4,) 
        elif 0.8<=np.random.rand()<0.9:
            bifur = -0.05 + 0.05 * np.random.rand(200+4,) 
        else:
            bifur = np.zeros(200+4, )

        #只随机生成
        new_contours = basedGeometry2Contour(bifur, contours) # 随机边界变化

        #加入梯度
        #new_contours = Grad2Contour(bifur,contours,orgfn)
        cv2.drawContours(produceImageBifur, new_contours, -1, (0, 0, 0), -1) # 新边界图

        #GT_image 是根据contours画出来的

        GT_image,Bif_image,regTargeted = self.cal_Dice(produceImage,produceImageBifur)

        mask = Bif_image * 255
        
        orgimg = Image.fromarray(np.uint8(orgimg)).convert('RGB')
        mask = Image.fromarray(np.uint8(mask)).convert('RGB')

        mylabel= produceImage.copy()
        mylabel[np.where(mylabel <=128)]= 128 # OC[np.where(OC <=128)]
        
        mylabel= Image.fromarray(np.uint8(mylabel))#.convert('RGB')
        mylabel_binary= Image.fromarray(np.uint8(GT_image*255))#.convert('RGB')
        GT= Image.fromarray(np.uint8(GT))#.convert('RGB')
    
        #mylabel_contours_400 = torch.from_numpy(creatContour(contours, 400))

        sample = {'image': orgimg,
                'GT':GT,
                'mylabel':mylabel,
                'mylabel_binary':mylabel_binary,
                'part':self.part,
                'train':self.Train,
                'mask':mask,
                'regTargeted':regTargeted,
                'imagename':imagename}
        
        #sample['mylabel_contours'] = mylabel_contours_400
       
        return self.transform_sample(sample)
       

    def __len__(self):
        return len(self.imgs)

    def cal_threshold(self,gray,part):
        if part == 'OD':
            gray[np.where(gray==0)]= 200 
            ret, threshold = cv2.threshold(gray, 200, 200, 0)
        if part == 'OC':
            gray[np.where(gray >=120 )] = 255
            ret, threshold = cv2.threshold(gray, 100, 255,cv2.THRESH_BINARY)  
        return ret, threshold

    def get_contours_from_image(self,GT,part):
        gray = GT.copy() #pix [255,128,0]
        #gray = cv2.cvtColor(gray,cv2.COLOR_BGR2RGB)
        #gray = gray[:,:,0]

        ret, threshold = self.cal_threshold(gray,part) #get res& threshold 
        contours2 = []
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for n in range(len(contours)):
            cnt = contours[n]
            area = cv2.contourArea(cnt)
            contours2.append([cnt,area])

        assert len(contours2)>0
        contours2 = sorted(contours2, key=lambda x:x[1])
        ct = np.array([contours2[0][0]])
        ct2 = ct[:,:,:,:]
        contours = ct2
        return contours

    def cal_IoU(self,GT_binary,mask_binary):
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

    def cal_Dice(self,GT_binary,mask_binary):
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
        regTargeted= 2*float(merge)/(combine+merge+eps)

        regTargeted_top = 2*float(merge)
        regTargeted_bottom = combine+merge+eps
        regTargeted_top = np.array([regTargeted_top])
        regTargeted_top = torch.from_numpy(regTargeted_top)
        regTargeted_bottom = np.array([regTargeted_bottom])
        regTargeted_bottom = torch.from_numpy(regTargeted_bottom)

        regTargeted = np.array([regTargeted])
        regTargeted = torch.from_numpy(regTargeted)
        return GT_image,Bif_image,[regTargeted_top,regTargeted_bottom]

    def transform_train(self,sample):
        composed_transforms = transforms.Compose([
            #distort.PhotometricDistort(),
            transfor.FixedResize(size=256),
            transfor.RandomHorizontalFlip(),
            #transfor.RandomScaleCrop(base_size=256, crop_size=256, min_ratio = 0.7, max_ratio = 1.3, fill=255),
            transfor.FixedResize(size=256),
            transfor.RandomGaussianBlur(),
            transfor.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transfor.REFUGE_norm(),
            transfor.ToTensor()])
        
        return composed_transforms(sample)
        #mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    def transform_val(self,sample):
        composed_transforms = transforms.Compose([
            transfor.FixedResize(size=256),
            transfor.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transfor.REFUGE_norm(),
            transfor.ToTensor()])

        return composed_transforms(sample)


# 随机选取label图像
class MyTestDataset(Dataset):
    def __init__(self, root, Train,loader=default_loader,part = 'OD'):
        #imgs = []
        imgs = os.listdir(os.path.join(root, "image"))
        labelimgs = os.listdir(os.path.join(root, "seg"))
        self.imgs = imgs
        self.labelimgs = labelimgs
        self.root = root
        self.Train = Train
        self.transform_sample = self.transform_train if self.Train else self.transform_val
        self.loader = loader
        self.part = part
        print('Train= {}, part = {}'.format(self.Train,self.part))
    def __getitem__(self, index):
        imagename = self.imgs[index]
        imgdir  = os.path.join(self.root, "image")
        orgfn  = os.path.join(imgdir, imagename)
        ##原图和标记图
        GT = Image.open(os.path.join(os.path.join(self.root, "seg"), imagename[:-3]+'bmp'))
        GT_rand = Image.open(os.path.join(os.path.join(self.root, "hanet_seg"), imagename.split('_')[-1][:-3]+'png'))
        randimagename = imagename
        orgimg = Image.open(orgfn).convert('RGB')
        orgimg = orgimg.resize((256,256))
        GT = GT.resize((256,256))
        GT_rand = GT_rand.resize((256,256))
        orgimg = np.array(orgimg)
        GT = np.array(GT)
        GT_rand = np.array(GT_rand)
        
        GT_contours = self.get_contours_from_image(GT,self.part)
        rand_contours = self.get_contours_from_image(GT_rand,self.part)
    
        emptyImage = GT.copy()
        emptyImage[:,:] = 255

        produceImage = emptyImage.copy()
        cv2.drawContours(produceImage, GT_contours, -1, (0, 0, 0), -1)
        produceImageBifur = emptyImage.copy()
        cv2.drawContours(produceImageBifur, rand_contours, -1, (0, 0, 0), -1) # 新边界图
        
        GT_image,Bif_image,regTargeted = self.cal_Dice(produceImage,produceImageBifur)

        mask = Bif_image * 255
        orgimg = Image.fromarray(np.uint8(orgimg)).convert('RGB')
        mask = Image.fromarray(np.uint8(mask)).convert('RGB')

        #真实label
        mylabel= produceImage.copy()
        mylabel[np.where( mylabel <=128 )]= 128
        
        mylabel = Image.fromarray(np.uint8(mylabel))#.convert('RGB')
        mylabel_binary= Image.fromarray(np.uint8(GT_image*255))#.convert('RGB')

        mylabel_rand= produceImageBifur.copy()
        mylabel_rand[np.where(mylabel_rand <=128 )]= 128
        mylabel_rand= Image.fromarray(np.uint8(mylabel_rand))#.convert('RGB')
        mylabel_rand_binary= Image.fromarray(np.uint8(Bif_image*255))#.convert('RGB')

        GT_rand= Image.fromarray(np.uint8(GT_rand))#.convert('RGB')
        GT = Image.fromarray(np.uint8(GT))#.convert('RGB')
        
        sample = {'image': orgimg,
            'GT':GT,
            'GT_rand':GT_rand,
            'mylabel':mylabel,
            'mylabel_binary':mylabel_binary,
            'mylabel_rand':mylabel_rand,
            'mylabel_rand_binary':mylabel_rand_binary,
            'part':self.part,
            'train':self.Train,
            'mask':mask,
            'regTargeted':regTargeted,
            'imagename':imagename}
        return self.transform_sample(sample)

    def __len__(self):
        return len(self.imgs)

    def cal_threshold(self,gray,part):
        if part == 'OD':
            gray[np.where(gray==0)]= 200 
            ret, threshold = cv2.threshold(gray, 200, 200, 0)
        if part == 'OC':
            gray[np.where(gray >=120 )] = 255
            ret, threshold = cv2.threshold(gray, 100, 255,cv2.THRESH_BINARY)  
        return ret, threshold

    def get_contours_from_image(self,GT,part):
        gray = GT.copy()
        #gray =  cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
        #gray  = gray[:,:,0]

        ret, threshold = self.cal_threshold(gray,part) #get res& threshold 
        contours2 = []
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for n in range(len(contours)):
            cnt = contours[n]
            area = cv2.contourArea(cnt)
            contours2.append([cnt,area])

        assert len(contours2)>0
        contours2 = sorted(contours2, key=lambda x:x[1])
        ct = np.array([contours2[0][0]])
        ct2 = ct[:,:,:,:]
        contours = ct2
        return contours

    def cal_IoU(self,GT_binary,mask_binary):
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

    def cal_Dice(self,GT_binary,mask_binary):
        #GT_image =  cv2.cvtColor(GT_binary, cv2.COLOR_BGR2RGB)
        GT_image = GT_binary.copy()
        GT_image[np.where(GT_image<255)] = 1
        GT_image[np.where(GT_image==255)] = 0
        GT_image_01 = GT_image#[:,:,0]
        #Bif_image 震荡之后的图片
        Bif_image =  cv2.cvtColor(mask_binary, cv2.COLOR_BGR2RGB)
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

    def transform_train(self,sample):
        composed_transforms = transforms.Compose([
            transforRand.FixedResize(size=256),
            transforRand.RandomHorizontalFlip(),
            #transforRand.RandomScaleCrop(base_size=256, crop_size=256, min_ratio = 0.7, max_ratio = 1.3, fill=255),
            transforRand.FixedResize(size=256),
            transforRand.RandomGaussianBlur(),
            transforRand.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforRand.REFUGE_norm(),
            transforRand.ToTensor()])
        
        return composed_transforms(sample)
        #mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    def transform_val(self,sample):
        composed_transforms = transforms.Compose([
            transforRand.FixedResize(size=256),
            transforRand.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforRand.REFUGE_norm(),
            transforRand.ToTensor()])

        return composed_transforms(sample)

class MemDataset(Dataset):
    def __init__(self, orgImage, oldcontours, pos, contours, transform):
        self.orgImage = orgImage
        #self.labelImage = labelImage
        self.pos = pos
        self.contours = contours
        self.oldcontours = oldcontours
        self.transform = transform

    def __getitem__(self, index):
        ##原图和标记图
        #labelimg = self.labelImage
        orgimg = self.orgImage
        #outlabelimg = self.outLabelImage
        contours = self.contours
        oldcontours = self.oldcontours
        pos = self.pos

        emptyImage = orgimg.copy()
        emptyImage[:,:,:] = 255

        produceImage = emptyImage.copy()
        cv2.drawContours(produceImage, oldcontours, -1, (0, 0, 0), -1) # 旧边界图

        contours = basedGeometry2Contour(pos, contours) # 边界变化

        produceImageBifur = emptyImage.copy()
        cv2.drawContours(produceImageBifur, contours, -1, (0, 0, 0), -1) # 新边界图

        GT_image =  cv2.cvtColor(produceImage, cv2.COLOR_BGR2RGB)
        GT_image[np.where(GT_image<255)] = 1
        GT_image[np.where(GT_image==255)] = 0
        GT_image_01 = GT_image[:,:,0]

        Bif_image =  cv2.cvtColor(produceImageBifur, cv2.COLOR_BGR2RGB)
        Bif_image[np.where(Bif_image<255)] = 1
        Bif_image[np.where(Bif_image==255)] = 0
        Bif_image_01 = Bif_image[:,:,0]

        # print(Bif_image_01.shape)
        mix_image = GT_image_01 + Bif_image_01
        # print((Bif_image_01==GT_image_01).all())
        ##获得交/并的回归值
        eps = 1e-7
        merge  = np.where(mix_image==2)[0].shape[0]
        combine = np.where(mix_image>0)[0].shape[0]
        regTargeted= float(merge)/(combine+eps)
        # print("回归值", regTargeted)

        orgimg[:,:,0] = Bif_image_01 * 255
        # orgimg = cv2.cvtColor(orgimg,cv2.COLOR_BGR2RGB)

        orgimg = Image.fromarray(np.uint8(orgimg)).convert('RGB')
        #labelimg = Image.fromarray(np.uint8(outlabelimg)).convert('RGB')
        orgimg = self.transform(orgimg)
        #labelimg = self.transform(labelimg)

        regTargeted =  np.array([regTargeted])
        regTargeted = torch.from_numpy(regTargeted)

        return orgimg, regTargeted # 原始图像转换B通道，标记图像，交/并的回归值

    def __len__(self):
        return 1

class MemTestDataset(Dataset):
    def __init__(self, orgImage, pos, contours, transform):
        self.orgImage = orgImage
        #self.labelImage = labelImage
        self.pos = pos
        self.contours = contours
        self.transform = transform

    def __getitem__(self, index):
        ##原图和标记图
        #labelimg = self.labelImage
        orgimg = self.orgImage
        #outlabelimg = self.outLabelImage
        contours = self.contours
        pos = self.pos

        emptyImage = orgimg.copy()
        emptyImage[:,:,:] = 255

        contours = basedGeometry2Contour(pos, contours) # 边界变化

        produceImageBifur = emptyImage.copy()
        cv2.drawContours(produceImageBifur, contours, -1, (0, 0, 0), -1) # 新边界图

        Bif_image =  cv2.cvtColor(produceImageBifur, cv2.COLOR_BGR2RGB)
        Bif_image[np.where(Bif_image<255)] = 1
        Bif_image[np.where(Bif_image==255)] = 0
        Bif_image_01 = Bif_image[:,:,0]


        orgimg[:,:,0] = Bif_image_01 * 255

        orgimg = Image.fromarray(np.uint8(orgimg)).convert('RGB')
        #labelimg = Image.fromarray(np.uint8(outlabelimg)).convert('RGB')
        orgimg = self.transform(orgimg)

        return orgimg # 原始图像转换B通道

    def __len__(self):
        return 1



if __name__ =='__main__':
    path_train = '/datastore/wait_to_review/sunyz/data/REFUGE_cut/train'
    path_test = '/datastore/wait_to_review/sunyz/data/REFUGE_cut/val'

    #dataset = MyDataset(path_train, True,part='OC')
    dataset = MyTestDataset(path_test,Train = False,part='OD')
    loader = torch.utils.data.DataLoader(dataset, batch_size=2,
                                              shuffle=True,num_workers=2)

    # testset = MyTestDataset(path_test, transform=mytransform,part='OC')

    # loader = torch.utils.data.DataLoader(testset, batch_size=1,
    #                                           shuffle=True)# , num_workers=4

    # testset2 = MyTestDataset(path_test, transform=mytransform,part='OC')
    # loader = torch.utils.data.DataLoader(testset2, batch_size=1,
    #                                           shuffle=False, num_workers=1)


    #from ..util import get_img_contours,tensor2im
    
   
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

    

    uploader = transforms.ToPILImage()
    for batch_idx,sample in enumerate(loader):

        img = sample['image']
        GT = sample['GT']
        GT_rand = sample['GT_rand']
        mylabel = sample['mylabel'] 
        mylabel_binary = sample['mylabel_binary']
        mylabel_rand = sample['mylabel_rand'] 
        mylabel_rand_binary = sample['mylabel_rand_binary']
        imagename = sample['imagename']
        part = sample['part']
        train = sample['train'] 
        mask = sample['mask'] 
        regTargeted = sample['regTargeted']
        print(regTargeted[0].data)
        torch.set_printoptions(threshold=np.inf)
        #print(img.size(),GT.size(),mylabel.size(),mylabel_binary.size(),mylabel_rand.size(),mylabel_rand_binary.size(),mask.size())
        img = torch.squeeze(img[0],0)
        img = tensor2im(img)
        cv2.imwrite('img.jpg',img)
        mask = torch.squeeze(mask[0],0)
        mask = tensor2im(mask)
        cv2.imwrite('mask.jpg',mask)
        mylabel_binary = torch.squeeze(mylabel_binary[0],0)
        utils.save_image(mylabel_binary, 'mylabel_binary.bmp', normalize=True)

        #mylabel_rand_binary = torch.squeeze(mylabel_rand_binary[0],0)
        #utils.save_image(mylabel_binary, 'mylabel_rand_binary.bmp', normalize=True)

        GT = torch.squeeze(GT[0],0)
        utils.save_image(np.abs(128.0*GT-255.), 'GT.bmp', normalize=True)

        GT_rand = torch.squeeze(GT_rand[0],0)
        utils.save_image(np.abs(128.0*GT_rand-255.), 'GT_rand.bmp', normalize=True)

    
        mylabel = torch.squeeze(mylabel[0],0)
        utils.save_image(np.abs(128.0*mylabel-255.), 'mylabel.bmp', normalize=True)

        mylabel_rand = torch.squeeze(mylabel_rand[0],0)
        utils.save_image(np.abs(128.0*mylabel_rand-255.), 'mylabel_rand.bmp', normalize=True)

        

        # final_matrix = np.zeros((256, 256*5, 3), np.uint8)
        # final_matrix[0:256, 0:256] = img
        # final_matrix[0:256, 256:256*2] = GT
        # final_matrix[0:256, 256*2:256*3] = mylabel
        # final_matrix[0:256, 256*3:256*4] = mylabel_binary
        # final_matrix[0:256, 256*4:256*5] = mask

        # final_matrix[0:256, 256*2:256*3] = GT_rand
        # final_matrix[0:256, 256*5:256*6] = mylabel_rand
        # final_matrix[0:256, 256*6:256*7] = mylabel_rand_binary
        #cv2.imwrite('111.png',final_matrix)
        
        print('write Done')
       
        break
        # print('bs_idx,name',imagename)

      

        # ODImgblackt = ODImgblack[0,0,:,:]
        # for i in range(ODImgblackt.size(0)):
        #     for j in range(ODImgblackt.size(1)):
        #         dic[ODImgblackt[i,j].data.item()] = ODImgblackt[i,j].data.item()

        # for k,v in dic.items():
        #     print(v)

        # ODImgblack[ODImgblack>=0.5] = 1
        # ODImgblack[ODImgblack<0.5] = 0
        # ODImgblack = ODImgblack[0,0,:,:].view(1,256,256)

        # uploader = transforms.ToPILImage()
        # ODImgblack = tensor2im_black(ODImgblack)
       
        #uploader(ODImgblack).save('111.png')
        #cv2.imwrite('11.png',img)
        # plt.imshow(img)
        # plt.show()
        # seg = get_img_contours(label,'OC')
        # print(seg.shape)
       
      
