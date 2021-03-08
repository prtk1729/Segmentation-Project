from wiener import wiener_driver, read_n_convert2gray
from aniso_diff import aniso_driver
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import find_peaks


def aprxTRL(path, idx):
    img = read_n_convert2gray(path, gb=False)
    original_gray_img = np.copy(img)

    # plt.imshow(img, cmap='gray')
    # plt.show()


    for i in range(11):
        filtered_img = wiener_driver(img, plot=False)
        imgout = aniso_driver(filtered_img, plot=False)
        img = imgout

    # ========= Apply Gaussian Kernel ===============================================================================
    def gaussian_2_remove_local_noise(img):
        '''Takes W+AD filtered_image and spits out the smooth gaussian image'''
        img = cv2.GaussianBlur(img, (11,11), 0.5) #1st param is a gray_scale_img, (kernel_size), 3rd param if set to 0
        return img

    img = gaussian_2_remove_local_noise(img)
    # ========= Apply Gaussian Kernel ===============================================================================


    # ==================
    def plot_intensity_profile_grid(patch, gl_max1, gl_max2):
        fig = plt.figure(figsize=(15,6))
        sub_fig_label = [f'a_scan_{i}' for i in range(patch.shape[0])]
        print('\n\nHi\n\n')
        for i in range(len(sub_fig_label)): 
            fig.add_subplot( 4, 5 , i+1)
            plt.plot(patch[i, :])
            plt.scatter(x = gl_max1[i], y = patch[i, gl_max1[i]], c='r', s=10)
            plt.scatter(x = gl_max2[i], y = patch[i, gl_max2[i]], c='r', s=10)

            # plt.gca().invert_yaxis()
            plt.grid(False)
            # plt.title(sub_fig_label[i])

        plt.show()
    # ================

    def associate_y_coords(gl_max1, gl_max2):
        y_coords1, y_coords3 = [], []

        for i in range(len(gl_max1)):
            if gl_max1[i] < gl_max2[i]:
                # associate correctly
                y_coords1.append(gl_max1[i])
                y_coords3.append(gl_max2[i])

            else:
                y_coords1.append(gl_max2[i])
                y_coords3.append(gl_max1[i])

        return y_coords1, y_coords3


    def find_2nd_peaks(img, y_coords3):
        '''Uptil y_coords then upper-bd
        b-scan max-peak investigate ==>plot if wrong 
        Add constraints'''

        patch = img
        gl_max2 = []
        for i in range(patch.shape[1]): 
            a_scan_line = patch[: y_coords3[i]-50, i]    
            peaks = find_peaks(a_scan_line, height= 0)
            max2 = np.array(peaks[1]['peak_heights']).argsort()[-2] 

            # print(peaks[0][req_idx] ) 
            gl_max2.append(peaks[0][max2])

        print(f'\n{len(gl_max2)}\n',len(gl_max2))
        return gl_max2


  # Find min1
    def find_min1(a_scan_max1, a_scan_line):
        '''Find just before local minima'''

        rnfl_coord = a_scan_max1
        k = -1
        a_scan_max1 -= 2
        while( a_scan_max1 ):
            # go back
            if( a_scan_line[ a_scan_max1] < a_scan_line[ (a_scan_max1-1) ]):
                # 1st local_minima => stop
                return a_scan_max1
            a_scan_max1 -= 1
        return 0




    # ============================= Find Max, 2nd Max, Min in one A-scan ============================================
    def find2max1min(img):
        from scipy.signal import find_peaks
        print('\n\nmax_coords', img.argmax( axis=0).shape )
        y_coords1, y_coords2, y_coords3 = [], [], []


        gl_max1, gl_max2, gl_min1 = [], [], []
        patch = img[:, :]   

        print('\n\nPatch_shape',patch.shape[0])
        for i in range(patch.shape[1]): 

            # max1
            a_scan_line = patch[:, i]    
            peaks = find_peaks(a_scan_line, height= 0)
            max1 = np.array(peaks[1]['peak_heights']).argsort()[-1]
            # print(peaks[0][req_idx] ) 
            gl_max1.append(peaks[0][max1])

            # min1
            min1 = find_min1(peaks[0][max1], a_scan_line )
            # thresh = 10
            # if( i >= 1 and ((min1 - gl_min1[-1]) > thresh) ):
            #     # smooth by updating
            #     min1 = gl_min1[-1]
            gl_min1.append(min1)


            # max2
            # print('max1',peaks[0][max1])
            peaks1 = find_peaks(patch[ :min1-10, i], height= 0)
            max2 = np.array(peaks1[1]['peak_heights']).argsort()[-1] 
            gl_max2.append(peaks1[0][max2] )


    
        y_coords1, y_coords3 = associate_y_coords(gl_max1, gl_max2)
        # print(y_coords1,'\n')
        # print(y_coords3, '\n')

        # smooth min1
        
        y_coords2 = gl_min1

        print(y_coords2, '\n')
        # plot_intensity_profile_grid(patch, gl_max1, gl_max2)
        # print('Hi',len(y_coords2), len(y_coords3), len(y_coords1) )

        return y_coords1, y_coords2, y_coords3

    y_coords1, y_coords2, y_coords3 = find2max1min(img)
    # ===========================

    def plot_grid_once_and_for_all(original_gray_img, img, scatter=False, 
            y_coords3=[], y_coords2=[], y_coords1=[]):
        # define fig.
        fig = plt.figure(figsize=(6,6))
        sub_fig_list = [original_gray_img, img]
        sub_fig_label = [f'Original_Gray_Scaled_{idx}', f'After_k_iters_{idx}']

        if scatter == True:
            implot = plt.imshow(img, cmap='gray')

            # put a red dot, size 40, at 2 locations:
            x = list(range(img.shape[1]))
            # print(list(y_coords1) )
            # print(len(x), y_coords1.shape)
            print(y_coords1)
            plt.scatter(x=x, y=list(y_coords3), c='r', s=0.1)
            # plt.scatter(x=x, y=list(y_coords2), c='g', s=0.1)
            plt.scatter(x=x, y=list(y_coords1), c='b', s=0.1)

        else:
            for i in range(len(sub_fig_list)):
                
                fig.add_subplot( 1, 2 , i+1)
                plt.imshow(sub_fig_list[i], cmap='gray')
                plt.title(sub_fig_label[i])

        plt.show()


    plot_grid_once_and_for_all(  original_gray_img,  img,  scatter=True, 
                                y_coords3=y_coords3, y_coords2=y_coords2, y_coords1=y_coords1 )
    


if __name__ == '__main__':
    paths = glob.glob('/Users/prateek/Desktop/Segmetation_github/Segmentation-Project/central_scan_task/data_files/OCT_4_sample_folders/2675931_21017_0_0/*.png')
    for i in range(3):
        aprxTRL(paths[i], i)
