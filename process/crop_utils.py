import os, sys
import numpy as np

# Import files from the local folder
root_path_ = os.path.abspath('.')
sys.path.append(root_path_)
from config import configuration


def get_partition_height(full_size_height):
    ''' Get the partition height
    Args:
        full_size_height (int):     The input full size image height
    Returns:
        partition_height (int):     The partition height
    '''
    
    partition_height = (full_size_height//3) + (configuration.pixel_padding//3) + configuration.pixel_padding

    return partition_height



def crop4partition(img, position = 3):
    ''' Crop img to THREE parition based on the setting from configuration
    Args:
        img (numpy):        The input full size image
        position (int):     An optional position index for which crop we need to return
    Returns:
        crop1, crop2, crop3 (numpy):   The cropped image
    '''

    # Preparation
    height, width, _ = img.shape
    crop_base = height // 3  # 160 for 480p
    side_extra_padding = configuration.pixel_padding // 3
    pixel_padding = configuration.pixel_padding
 

    # Crop the images with pixel padding ad adjust
    if position == 0:
        crop1 = img[:crop_base + side_extra_padding+pixel_padding, :, :] # :168
        return crop1

    elif position == 1:
        crop2 = img[crop_base+side_extra_padding-pixel_padding : 2*crop_base-side_extra_padding+pixel_padding, :, :] # 156:324
        return crop2

    elif position == 2:
        crop3 = img[2*crop_base-side_extra_padding-pixel_padding:, :, :] # 312:480
        return crop3

    elif position == 3:
        crop1 = img[:crop_base + side_extra_padding+pixel_padding, :, :] # :168
        crop2 = img[crop_base+side_extra_padding-pixel_padding : 2*crop_base-side_extra_padding+pixel_padding, :, :] # 156:324
        crop3 = img[2*crop_base-side_extra_padding-pixel_padding:, :, :] # 312:480
        return (crop1, crop2, crop3)
    
    else:
        raise NotImplementedError("This position crop is not supported!")


def crop4partition_SR(img, position = 3):
    ''' Crop img to THREE parition based on the setting from configuration FOR SR version (need scale info)
    Args:
        img (numpy):        The input full size image
        position (int):     An optional position index for which crop we need to return
    Returns:
        crop1, crop2, crop3 (numpy):   The cropped image
    '''

    # Preparation
    scale = configuration.scale
    height, width, _ = img.shape
    crop_base = height // 3  # 160 for 480p
    side_extra_padding = configuration.pixel_padding // 3
    pixel_padding = configuration.pixel_padding
 

    # Crop the images with pixel padding ad adjust
    if position == 0:
        crop1 = img[:crop_base + scale*(side_extra_padding+pixel_padding), :, :] # : 168*scale
        return crop1

    elif position == 1:
        crop2 = img[crop_base + scale*(side_extra_padding-pixel_padding) : 2*crop_base + scale*(-side_extra_padding+pixel_padding), :, :] # 156*scale : 324*scale
        return crop2

    elif position == 2:
        crop3 = img[2*crop_base + scale*(-side_extra_padding-pixel_padding):, :, :] # 312*scale : 480*scale
        return crop3

    elif position == 3:
        crop1 = img[:crop_base + side_extra_padding+pixel_padding, :, :] # :168
        crop2 = img[crop_base+side_extra_padding-pixel_padding : 2*crop_base-side_extra_padding+pixel_padding, :, :] # 156:324
        crop3 = img[2*crop_base-side_extra_padding-pixel_padding:, :, :] # 312:480
        return (crop1, crop2, crop3)
    
    else:
        raise NotImplementedError("We didn't find such position of the frame ")


def combine_partitions_SR(crop1, crop2, crop3):
    ''' Combine three cropped frames to a big one and crop uncessary part
    Args:
        crop1, crop2, crop3 (numpy):   The cropped image
    Returns:
        img (numpy):        The input full size image
    '''
    assert(crop1.shape == crop2.shape)
    assert(crop3.shape == crop2.shape)


    scale = configuration.scale
    cropped_padding = configuration.pixel_padding
    h, w, c = crop1.shape

    # Crop
    crop1_ = crop1[ : -scale*cropped_padding,  :,:]          
    crop2_ = crop2[scale*cropped_padding : -scale*cropped_padding,  :,:]
    crop3_ = crop3[scale*cropped_padding:,  :,:]


    # The Cropped frame don't need to have the same shape (crop2_ is smaller)
    combined_frame = np.concatenate((crop1_, crop2_, crop3_))

    return combined_frame