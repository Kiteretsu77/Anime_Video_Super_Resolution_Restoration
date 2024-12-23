# The first three need to import from the following order; else, a bug may appear
import tensorrt
from torch2trt import torch2trt
import torch 

from torch import nn as nn
from torch.nn import functional as F
from torch2trt import TRTModule
from torchvision.transforms import ToTensor

from time import time as ttime
import cv2
import numpy as np
import time
import os, sys
from time import time as ttime
import argparse
import shutil
import requests


# Import from local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from config import configuration
from process.utils import np2tensor
from process.crop_utils import get_partition_height



class Generator:

    def __init__(self, input_dir):
        if not os.path.exists(input_dir):
            print("no such sample_input_dir exists: ", input_dir)
            os._exit(0)

        if not os.path.exists(input_dir):
            print("no sample input path {} exists for tensorrt weight generation ".format(input_dir))
            os._exit(0)

        # Read image info
        img = cv2.imread(input_dir)
        self.sample_input = np.array(img)
        self.h, self.w, _ = img.shape
        print("TensorRT weight Generator will process the image with height {} and width {} ".format(self.h, self.w))
        

        # Other config setup
        self.dont_calculate_transform = None


    def cunet_pre_process(self, array):
        # Don't forget this np2tensor
        tensor = np2tensor(array, pro=True)
        # tensor = ToTensor()(array).unsqueeze(0).cuda()
        input = F.pad(tensor, (18, 18, 18, 18), 'reflect').cuda()  # (18, 18, 18, 18) is hard-code padding

        return input
    
    def rrdb_preprocess(self, array):
        # RRDB just directly use ToTensor, which is different from cunnet preprocess
        tensor = np2tensor(array, pro=False)

        return tensor


    def model_weight_transform(self, input):

        # Preparation
        torch.cuda.empty_cache()


        # Load weight
        if not os.path.exists(self.org_weight_store_path):
            print("{} does not exist!".format(self.org_weight_store_path))
            os._exit(0)
        model_weight = torch.load(self.org_weight_store_path)

        # Load the model and prepare input case by case
        if configuration.architecture_name == "cunet":
            # Process CuNet (Real-CuGAN)
            from Real_CuGAN.cunet import UNet_Full
            generator = UNet_Full()
            input = self.cunet_pre_process(input)
            if "pro" in model_weight:
                # We need to delete "pro" part in cunet (Real-CUGAN)
                del model_weight["pro"]

        elif configuration.architecture_name == "rrdb":
            # Process RRDB (Real-ESRGAN)
            from Real_ESRGAN.rrdb import RRDBNet
            generator = RRDBNet(scale=configuration.scale)
            
            if configuration.model_name == "Real-ESRGAN":
                model_weight = model_weight['params_ema']

            elif configuration.model_name == "VCISR":
                model_weight = model_weight['model_state_dict']

            input = self.rrdb_preprocess(input)

        else:
            print("We don't support this architecture ", configuration.architecture_name)
            raise NotImplementedError()     

        generator.load_state_dict(model_weight, strict=True)
        generator.eval().cuda()

        
        
        with torch.no_grad():
            if args.int8_mode:
                from calibration import ImageFolderCalibDataset
                print("Use int8 mode")
                mode = "int8"
                print("intput shape is ", input.shape)
                dataset = ImageFolderCalibDataset("imgs/", self.h, self.w)
                model_trt_model = torch2trt(generator, [input], int8_mode=True, int8_calib_dataset=dataset)
            else:
                print("Use float16 mode in TensorRT")
                mode = "float16"
                input = input.half() 
                generator = generator.half()
                print("Generating the TensorRT weight ........")
                model_trt_model = torch2trt(generator, [input], fp16_mode=True)

        # Store and Save
        save_path = os.path.join(self.tensorrt_weight_store_dir, 'trt_' + self.base_name + '_' + mode + '_weight.pth')
        torch.save(model_trt_model.state_dict(), save_path)
        print("Finish generating the tensorRT weight and save at {}".format(save_path))


        # Transform to trt model
        output = model_trt_model(input)
        print("TensorRT Sample input shape is ", input.shape)
        print("TensoRT Sample output shape is ", output.shape)

        return output
        

    def weight_generate(self):
        self.dont_calculate_transform = False

        self.model_weight_transform(self.sample_input)

        print("TensorRT weight is stored!")


    def run(self, partition = False):
        
        # some global setting
        self.base_name = str(int(self.w)) + "X" + str(int(self.h))
        self.org_weight_store_path =  os.path.join(configuration.weights_dir, configuration.model_name, configuration.architecture_name + "_weight.pth")
        self.tensorrt_weight_store_dir = os.path.join(configuration.weights_dir, configuration.model_name)

        ###################################################################################

        # Generate tensorrt weight
        self.weight_generate()



def generate_partition_frame(full_frame_dir):
    # Cut the frame to three fold for Video redundancy acceleartion in inference.py

    img = cv2.imread(full_frame_dir)
    h, _, _ = img.shape
    partition_height = get_partition_height(h)
    
    
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    partition_img = img[:partition_height, :,:]
    print("Partition Size (1in3) after crop is ", partition_img.shape)
    partition_frame_dir = configuration.partition_frame_dir
    cv2.imwrite(partition_frame_dir, partition_img)

    return partition_frame_dir


def crop_image(sample_img_dir, target_h, target_w):
    img = cv2.imread(sample_img_dir)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Check if image is over size
    h, w, _ = img.shape
    if h < target_h or w < target_w:
        print("Such height and/or width is not supported, please use a larger sample input")
        os._exit(0)

    croped_img = img[:target_h, :target_w,:] 
    print("Size after crop is ", croped_img.shape)
    cv2.imwrite(configuration.full_croppped_img_dir, croped_img)


def tensorrt_transform_execute(target_h, target_w, sample_img_dir=configuration.sample_img_dir):
    start = time.time()

    # Crop image to the desired height and width we need
    crop_image(sample_img_dir, target_h, target_w)

    # Generate full frame
    if not args.only_partition_frame:
        ins = Generator(configuration.full_croppped_img_dir) 
        ins.run()
        print("Full Frame weight generation Done!")


    # Generate partition frame
    if not args.only_full_frame:
        partition_img_dir = generate_partition_frame(configuration.full_croppped_img_dir)
        ins = Generator(partition_img_dir) 
        ins.run(partition = True)
        print("Partition Frame generation Done!")


    print("Total time spent on tensorrt weight generation is %d s" %(int(time.time() - start)))


def check_file():
    '''
        Check if the fundamental model weight exists; if not, download them
    '''

    # Define the location and the weight name for the saved weight
    weight_store_path = os.path.join(configuration.weights_dir, configuration.model_name, configuration.architecture_name + "_weight.pth")

    if not os.path.exists(weight_store_path):
        print("There isn't " + weight_store_path + " under weights folder")
        network_url = {
            "Real-CUGAN" : "https://drive.google.com/u/0/uc?id=1hc1Xh_1qBkU4iGzWxkThpUa5_W9t7GZ_&export=download",
            "Real-ESRGAN" : "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "VCISR": "https://drive.google.com/u/0/uc?id=1y59iJ86xujt3990tzyyBAq73VcPKJ3ri&export=download",
        }
        

        # Automatically download Code, but if you want other weight, like less denoise, please go see https://drive.google.com/drive/folders/1jAJyBf2qKe2povySwsGXsVMnzVyQzqDD
        print("We will automatically download pretrained weight of " + configuration.model_name + " !!!")

        # Download the content
        url = network_url[configuration.model_name]
        r = requests.get(url, allow_redirects=True)

        # Store the weight to the target path
        open(weight_store_path, 'wb').write(r.content)

        print("Finish downloading pretrained weight!")


def generate_weight(lr_h = 540, lr_width = 960):
    parse_args()
    check_file()


    if configuration.use_tensorrt:
        tensorrt_transform_execute(lr_h, lr_width)
        print(configuration.model_name + " tensorrt weight transforms has finished!")
    


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--only_full_frame", action='store_true', help="")
    parser.add_argument("--only_partition_frame", action='store_true', help="")
    parser.add_argument('--test_dir', type=str, default="", help=" If you want to have personal test input to calibrate, you can use this one.")
    parser.add_argument("--my_model", action='store_true', help="to load personal trainned model")
    # parser.add_argument("--int8_mode", action='store_true')   // Very unstable, so we don't recommend using it.
    

    global args
    args = parser.parse_args()

    args.int8_mode = False

if __name__ == "__main__":
    generate_weight()


