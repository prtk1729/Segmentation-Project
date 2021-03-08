import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import argparse


def read_n_convert2gray(path, gb=True):

    global paths
    # read the img
    img = cv2.imread(path)
    print(img.shape)

    # copy the img (recall best practice)
    img_copy = np.copy(img)

    # ColorSpace tranform
    img = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
    # plt.imsave(f'../out_files/{paths[0].split("/")[-1]}_original.png', img, cmap='gray')

    print(img.shape)

    if(gb == True):
        # GaussianBlur
        # img = cv2.GaussianBlur(img, (3,3), 0) #1st param is a gray_scale_img, (kernel_size), 3rd param if set to 0

        # Median Filter
        img = cv2.medianBlur(img, 3)

        # GaussianBlur
        img = cv2.GaussianBlur(img, (3,3), 0) #1st param is a gray_scale_img, (kernel_size), 3rd param if set to 0

        # means automatically finds the std. dev.
        # plt.imsave(f'../out_files/{paths[0].split("/")[-1]}_medianBlur->gb_(3,3).png', img, cmap='gray')

    return img


def perform_Canny_Edge_Detection_and_plot(gray, gb=True):
    global paths

    # gray = gray/255.0
    # define lower and upper thresh
    lower = 180
    upper = 240

    print(np.amax(gray))
    edges = cv2.Canny(gray, lower, upper)
    plt.imshow(edges, 'gray')
    plt.imsave(f'../out_files/{paths[0].split("/")[-1]}_mb{gb}->gb_ce(3,3).png', edges, cmap='gray')
    plt.show()

   



# paths = glob.glob('/Users/prateek/Desktop/Segmetation_github/Segmentation-Project/central_scan_task/data_files/OCT_4_sample_folders/2675931_21017_0_0/*.png')

# blurred_gray_img = read_n_convert2gray(paths[0], gb=True)
# print(blurred_gray_img.shape)
# perform_Canny_Edge_Detection_and_plot(blurred_gray_img, gb=True)
