import numpy as np
import sys

"""
Convolutional neural network implementation using NumPy.
An article describing this project is titled "Building Convolutional Neural Network using NumPy from Scratch". It is available in these links: https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad/
https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html
It is also translated into Chinese: http://m.aliyun.com/yunqi/articles/585741
The project is tested using Python 3.5.2 installed inside Anaconda 4.2.0 (64-bit)
NumPy version used is 1.14.0
For more info., contact me:
    Ahmed Fawzy Gad
    KDnuggets: https://www.kdnuggets.com/author/ahmed-gad
    LinkedIn: https://www.linkedin.com/in/ahmedfgad
    Facebook: https://www.facebook.com/ahmed.f.gadd
    ahmed.f.gad@gmail.com
    ahmed.fawzy@ci.menofia.edu.eg
"""

def conv_(img, conv_filter,padding):
    kernel_size = conv_filter.shape[1]
    
    #Looping through the image to apply the convolution operation.
    # if image size=y, kernel size=x , row =y-x+1
    ##this method doesn't consider the bording information(doesn't do the padding)
    if padding==False:
        result = np.zeros((img.shape))        
    else:
        img = np.pad(img, pad_width=int(kernel_size/2.0), mode='constant', constant_values=0)
        result = np.zeros((img.shape))
    for r in np.uint16(np.arange(kernel_size/2.0, 
                        img.shape[0]-kernel_size/2.0+1)):
        for c in np.uint16(np.arange(kernel_size/2.0, 
                                        img.shape[1]-kernel_size/2.0+1)):
            """
            Getting the current region to get multiplied with the filter.
            How to loop through the image and get the region based on 
            the image and filer sizes is the most tricky part of convolution.
            """
            ##np.floor return int part of number. ceil no conditional to next bigger number
            curr_region = img[r-np.uint16(np.floor(kernel_size/2.0)):r+np.uint16(np.ceil(kernel_size/2.0)), 
                            c-np.uint16(np.floor(kernel_size/2.0)):c+np.uint16(np.ceil(kernel_size/2.0))]
            #Element-wise multipliplication between the current region and the filter.
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result) #Summing the result of multiplication.
            result[r, c] = conv_sum #Saving the summation in the convolution layer feature map.
        
    print("result shape=",result.shape)        
    #Clipping the outliers of the result matrix.
    final_result = result[np.uint16(kernel_size/2.0):result.shape[0]-np.uint16(kernel_size/2.0), 
                          np.uint16(kernel_size/2.0):result.shape[1]-np.uint16(kernel_size/2.0)]
    return final_result
def conv(img, conv_filter,padding):

    if len(img.shape) != len(conv_filter.shape) - 1: # Check whether number of dimensions is the same
        print("Error: Number of dimensions in conv filter and image do not match.")  
        exit()
    if len(img.shape) > 2 or len(conv_filter.shape) > 3: # Check if number of image channels matches the filter depth.
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()
    if conv_filter.shape[1] != conv_filter.shape[2]: # Check if filter dimensions are equal.
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
        sys.exit()
    if conv_filter.shape[1]%2==0: # Check if filter diemnsions are odd.
        print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
        sys.exit()

    # An empty feature map to hold the output of convolving the filter(s) with the image.
    if padding:
        feature_maps = np.zeros((img.shape[0],img.shape[1],conv_filter.shape[0]))
    else:
        feature_maps = np.zeros((img.shape[0]-conv_filter.shape[1]+1, 
                                img.shape[1]-conv_filter.shape[1]+1, 
                                conv_filter.shape[0]))

    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):
        print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :] # getting a filter from the bank.
        """ 
        Checking if there are mutliple channels for the single filter.
        If so, then each channel will convolve the image.
        The result of all convolutions are summed to return a single feature map.
        """
        if len(curr_filter.shape) > 2:
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0],padding) # Array holding the sum of all feature maps.
            for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.
                conv_map = conv_map + conv_(img[:, :, ch_num], 
                                  curr_filter[:, :, ch_num],padding)
        else: # There is just a single channel in the filter.
            conv_map = conv_(img, curr_filter,padding)
        feature_maps[:, :, filter_num] = conv_map # Holding feature map with the current filter.
    return feature_maps # Returning all feature maps.
    

def pooling(feature_map, size=2, stride=2):
    #Preparing the output of the pooling operation.
    pool_out = np.zeros((np.uint16((feature_map.shape[0]-size+1)/stride+1),
                            np.uint16((feature_map.shape[1]-size+1)/stride+1),
                            feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0,feature_map.shape[0]-size+1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1]-size+1, stride):
                pool_out[r2, c2, map_num] = np.max([feature_map[r:r+size,  c:c+size, map_num]])
                c2 = c2 + 1
            r2 = r2 +1
    return pool_out

def relu(feature_map):
    #Preparing the output of the ReLU activation function.
    relu_out = np.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in np.arange(0,feature_map.shape[0]):
            for c in np.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = np.max([feature_map[r, c, map_num], 0])
    return relu_out