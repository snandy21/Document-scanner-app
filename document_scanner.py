"""
##################################################
Document Analysis Project
Author : Suprojit Nandy
Affiliation : Hardware and Embedded Systems Lab, School of Computer Science & Engineering, NTU Singapore
##################################################
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import time

"""
Function Definition for extracting the pipeline.
We do the following :
1. Convert image to gray scale image
2. Binary thresholding inversion 
3. Extract the External Contours 
4. Check area wise to extract the largest contour
"""

def preprocessing_document(img):
    print(img.shape)
    print('The image is shown as:')
    # Rescale the image here :
    img_res  = cv2.resize(img, None,  fx = 0.5, fy= 0.5, interpolation=cv2.INTER_LINEAR)
    img_cp = img_res.copy()  # copy of the image stored here
    img_cp_scan = img_res.copy()
    # Change to gray scale here :
    img_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    img_gray_3_ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_binary, img_binary_inv = thresholded_vectorized(img_gray, thresh= 150, max_val=255)
    img_out = np.hstack((img_res, img_gray_3_ch)) # horizontal concatenation
    img_binarized_out = np.hstack((img_binary, img_binary_inv))
    # find the contours here :
    contours, hierarchy = cv2.findContours(img_binarized_out,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find only external contours here
    # Sort the contours as per their Areas :
    #mask = np.zeros(img_cp.shape, np.uint8)
    areas_max = sorted(contours, key=cv2.contourArea)
    # Show the contour Image here:
    #cv2.drawContours(img_cp, contours, -1, (0,255,0), 5)
    # store the document contour here :
    doc_contour = areas_max[-2]
    # store the width and height of the bounding rectangles of the contours here :
    x,y,width, height = cv2.boundingRect(areas_max[-2])
    print((width,height))
    # evaluate the 4 corners of the document here :
    # determine the most extreme points along the contour
    point_list1 = []
    Left_most = tuple(doc_contour[doc_contour[:, :, 0].argmin()][0])
    print(Left_most)

    Right_most = tuple(doc_contour[doc_contour[:, :, 0].argmax()][0])
    print(Right_most)

    Top_most = tuple(doc_contour[doc_contour[:, :, 1].argmin()][0])
    print(Top_most)

    Bot_most = tuple(doc_contour[doc_contour[:, :, 1].argmax()][0])
    print(Bot_most)
    point_list1 = np.float32([Left_most , Top_most , Right_most , Bot_most])
    print(point_list1)
    # Transformed Out Points :
    point_list2 = np.float32([[0,0],[width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    print(point_list2)
    # Perform Perspective Transformation :
    perspective = cv2.getPerspectiveTransform(point_list1, point_list2)
    img_scan_out = cv2.warpPerspective(img_cp_scan, perspective,(width, height))
    cv2.drawContours(img_cp, areas_max[-2], -1, (0,255,0), 5)
    cv2.circle(img_cp, Left_most, 5, (255,0,0),thickness= -1)
    cv2.circle (img_cp, Right_most, 5, (255,0,0),thickness= -1)
    cv2.circle(img_cp, Top_most, 5, (255,0,0), thickness= -1)
    cv2.circle(img_cp, Bot_most, 5, (255,0,0), thickness= -1)
    # Size of the Perspective transformed image here same as dimesnion of contour:
    image_sequence_results = np.hstack((img_res, img_cp))
    cv2.imshow('Document Resized and Contours established', image_sequence_results)

    #cv2.imshow('Iverted Images', img_binarized_out)
    cv2.waitKey(0)
    cv2.imshow('Scanned Output Document', img_scan_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def thresholded_vectorized(src, thresh, max_val):
    dst = np.zeros_like(src)
    dst2= np.zeros_like(src)
    threholded_up_pixels = src > thresh
    threholded_down_pixels = src < thresh
    dst[threholded_up_pixels] = max_val
    dst2[threholded_down_pixels] = max_val
    # Invert the pixel values as well :

    return dst, dst2

def main():
    # Read the image here :
    image = cv2.imread('scanned-form.jpg')
    img_cp = image.copy()
    preprocessing_document(image)

main()