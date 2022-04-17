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
                 sync_bn=False, hop=2, delta=0.5):
        super(HANet, self).__init__()
        

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        
       
      
        self.encoder = build_encoder(encoder, output_stride, BatchNorm)

        self.module = getattr(modules, module)(BatchNorm, delta=delta, hop=hop, stride=1)

        self.decoder = build_decoder(num_classes, encoder, BatchNorm)

        #print('hamodel finetune:',self.finetune_e )
    def forward(self, input):
       
        #output_e, low_level_feat = self.encoder(input)

        output_e, low_level_feat,l2,l3,l0 = self.encoder(input)

        output_mid = self.module(output_e)
        output_d = self.decoder(output_mid, low_level_feat)
        x = F.interpolate(output_d, size=input.size()[2:], mode='bilinear', align_corners=True)
        # score = output#.view(output.size(0),-1)
        # score = self.conv2linear(score).view(output.size(0),-1)
        # # v1 v2
        # score = self.score(score)
        #v3
        # score_t = self.score_1(score)
        # score_b = self.score_2(score)
        #return x,output_e,output_mid,output_d
        return x,output_e,output_mid,output_d,l0,low_level_feat,l2,l3
        
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
                            





