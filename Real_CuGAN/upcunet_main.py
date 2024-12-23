from torch2trt import TRTModule
from torch import nn as nn
import torch
from torch.nn import functional as F
from time import time as ttime
import numpy as np
import os, sys

# import files from local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from process.utils import np2tensor, tensor2np
from config import configuration
from Real_CuGAN.cunet import UNet_Full


class UpCunet2x(nn.Module):
    def __init__(self, unet_full_weight_path, adjust):
        super(UpCunet2x, self).__init__()
        load_start = ttime()


        torch.cuda.empty_cache()
        if configuration.use_tensorrt:
            self.unet_model_full = TRTModule()
            self.unet_model_full.load_state_dict(torch.load(unet_full_weight_path))
            # don't use .eval().cuda() because it will raise a bug
            for param in self.unet_model_full.parameters():
                param.grad = None
        else:
            # Use original pretrained model
            self.unet_model_full = UNet_Full()
            model_weight = torch.load(os.path.join(configuration.weights_dir, configuration.model_name, configuration.architecture_name+'_weight.pth'))
            if "pro" in model_weight:
                del model_weight["pro"]
            self.unet_model_full.load_state_dict(model_weight, strict=True)
            self.unet_model_full.eval().cuda()

        print("torch2trt unet full load+prepare time %.3f s"%(ttime() - load_start))



    def forward(self, input):
        x = F.pad(input, (18, 18, 18, 18), 'reflect')  # (18, 18, 18, 18) is emperical padding hard-code

        ######################## Neural Network Inference #############################
        unet_full_output = self.unet_model_full(x)
        #############################################################################


        # pro mode in default process
        return ((unet_full_output - 0.15) * (255/0.7)).round().clamp_(0, 255).byte()

        
    
class RealCuGAN_Scalar(object):
    def __init__(self, unet_full_weight_path, adjust):
        self.model = UpCunet2x(unet_full_weight_path, adjust)
        if configuration.use_tensorrt:
            self.model = self.model.half()          # Must use half in tensorrt for float16, else the output is a black screen

    def __del__(self):
        # if self.counter:
        #     print("Inner time is %.2f s on %d, which is %.5f s per frame"%(self.inner_times, self.counter, self.inner_times/self.counter))
        return


    def __call__(self, frame, position):
        
        tensor = np2tensor(frame, pro=True)
        if configuration.use_tensorrt:
            tensor = tensor.half()          # Must use half in tensorrt for float16, else the output is a black screen

        # Inference here
        res = self.model(tensor)

        result = tensor2np(res)
        return result