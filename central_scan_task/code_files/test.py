import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob

def read_n_convert2gray(path):

    # read the img
    img = cv2.imread(path)
    print(img.shape)

    # copy the img (recall best practice)
    img_copy = np.copy(img)

    # ColorSpace tranform
    img = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
    print(img.shape)
    return img


def perform_Canny_Edge_Detection_and_plot(gray):
    # define lower and upper thresh
    lower = 180
    upper = 240

    edges = cv2.Canny(gray, lower, upper)
    plt.imshow(edges, 'gray')
    plt.show()

paths = glob.glob('/Users/prateek/Desktop/alice_mnt_pt/2675931_21017_0_0/*.png')

gray_img = read_n_convert2gray(paths[0])
perform_Canny_Edge_Detection_and_plot(gray_img)
