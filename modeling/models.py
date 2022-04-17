import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling import modules
from modeling.encoder import build_encoder
from modeling.decoder import build_decoder


class LearnableSigmoid(nn.Module):
    def __init__(self, ):
        super(LearnableSigmoid, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.fill_(1.0)
        
    def forward(self, input):
        return 1/(1 +  torch.exp(-self.weight*input))
     
class myReLU(nn.Module):
    def __init__(self):
        super(myReLU,self).__init__()
    def forward(self,x):
        x = torch.clamp(0.5*x,min=-0.5)
        return x 

class HANet(nn.Module):
    def __init__(self, module ='HA_module', encoder='resnet101', output_stride=16, num_classes=2,
                 sync_bn=False, hop=2, delta=0.5,finetune = False):
        super(HANet, self).__init__()
        
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        
        #self.sigmoid = nn.Sigmoid()
        self.sigmoid = LearnableSigmoid()
        self.relu = myReLU()
        self.encoder = build_encoder(encoder, output_stride, BatchNorm)
        self.module = getattr(modules, module)(BatchNorm, delta=delta, hop=hop, stride=1)
        self.decoder = build_decoder(num_classes, encoder, BatchNorm)
        self.finetune = finetune
        if self.finetune:
            for p in self.parameters():
                p.requires_grad = False
        # self.score = nn.Sequential(*[
        #                             nn.Linear(256*16*16,1024),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(1024,512),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(512,256),
        #                             nn.ReLU(inplace=True),
        #                             nn.Linear(256,1),
        #                             self.relu
        #                             ])
        
       
        self.conv2linear = nn.Sequential(*[nn.Conv2d(256,256,kernel_size=3,stride=1),
                                nn.BatchNorm2d(256,affine=True),
                                nn.Conv2d(256,256,kernel_size=1,stride=1),
                                nn.BatchNorm2d(256,affine=True)])
        '''                    
        #v1
        self.score = nn.Sequential(*[
                                nn.Linear(256*14*14,2048),
                                nn.ReLU(inplace=True),
                                nn.Linear(2048,1024),
                                nn.ReLU(inplace=True),
                                nn.Linear(1024,512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512,256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256,128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128,64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64,1),
                                self.sigmoid
                                ])
         '''            
        # v2 
        self.score = nn.Sequential(*[
                                    nn.Linear(256*14*14,1024),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(1024,512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512,256),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(256,1),
                                    self.relu
                                    ])
        '''
        # v3
        self.score_1 = nn.Sequential(*[
                                    nn.Linear(256*14*14,1024),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(1024,512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512,256),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(256,1),
                                    self.relu
                                    ])
        self.score_2 = nn.Sequential(*[
                                    nn.Linear(256*14*14,1024),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(1024,512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512,256),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(256,1),
                                    self.relu
                                    ])
        '''
        self.finetune = finetune
        print('hamodel finetune:',self.finetune )
    def forward(self, input):
       
        x, low_level_feat = self.encoder(input)
        output = self.module(x)
        x = self.decoder(output, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        score = output#.view(output.size(0),-1)
        score = self.conv2linear(score).view(output.size(0),-1)
        # v1 v2
        score = self.score(score)
        #v3
        # score_t = self.score_1(score)
        # score_b = self.score_2(score)
        return x,score
        # return x,[score_t,score_b]


    def get_1x_lr_params(self):
        modules = [self.encoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.module, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
                            



