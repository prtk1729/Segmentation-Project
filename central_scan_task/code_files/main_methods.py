from wiener import wiener_driver, read_n_convert2gray
from aniso_diff import aniso_driver
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import statistics as stats
from test import read_n_convert2gray
from curve_fitting import curve_fitting
import json


def aprxTRL(path, idx, global_dict):
    img = read_n_convert2gray(path, gb=False)
    original_gray_img = np.copy(img)


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



    # ==================
    def plot_intensity_profile_grid(patch, gl_max1, gl_max2):
        fig = plt.figure(figsize=(15,6))
        sub_fig_label = [f'a_scan_{i}' for i in range(patch.shape[0])]
        for i in range(len(sub_fig_label)): 
            fig.add_subplot( 4, 5 , i+1)
            plt.plot(patch[i, :])
            plt.scatter(x = gl_max1[i], y = patch[i, gl_max1[i]], c='r', s=10)
            plt.scatter(x = gl_max2[i], y = patch[i, gl_max2[i]], c='r', s=10)

            plt.grid(False)
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


    def find3(img, y_min, y_max):
        '''
        Pass Sobel as img and Y-min and y_max for the layer
        '''
        sobelY = img
        # print('\n\nmax_coords', img.argmax( axis=0).shape )
        gl_max1, gl_max2, gl_max3 = [], [], []
        patch = sobelY[:, :] 

        y_coords1, y_coords2, y_coords3 = [], [], []
        for j in range(inp_edge_map.shape[1]):
            a_scan_line = patch[:, j] 
            # if 1st 5 px are > 0 ==> put the initial idx and break to next j
            for i in range(330):
                if a_scan_line[i] > 0:
                    # 1st non-zero
                    y_coords1.append(i)
                    break

        return y_coords1, y_coords2, y_coords3







    # ============================= Find Max, 2nd Max, Min in one A-scan ============================================
    def find3max(img, y_min, y_max):
        '''
        Pass Sobel as img and Y-min and y_max for the layer
        '''
        from scipy.signal import find_peaks
        print('\n\nmax_coords', img.argmax( axis=0).shape )
        gl_max1, gl_max2, gl_max3 = [], [], []
        patch = img[:, :] 

        for i in range(patch.shape[1]): 
            a_scan_line = patch[:, i]   

            peaks = find_peaks(a_scan_line, height= 0)
            max1 = np.array(peaks[1]['peak_heights']).argsort()[-1]
            gl_max1.append(peaks[0][max1])

            peaks1 = find_peaks(patch[ :, i], height= 0)
            max2 = np.array(peaks1[1]['peak_heights']).argsort()[-2] 
            gl_max2.append(peaks1[0][max2] )

            peaks1 = find_peaks(patch[ :, i], height= 0)
            max3 = np.array(peaks1[1]['peak_heights']).argsort()[-3] 
            gl_max3.append(peaks1[0][max3] )

        y_coords1, y_coords3 = associate_y_coords(gl_max1, gl_max2)
        y_coords1, y_coords2 = associate_y_coords(gl_max1, gl_max3)
        return y_coords1, y_coords2, y_coords3





    # ============================= Find Max, 2nd Max, Min in one A-scan ============================================
    def find2max1min(img):
        from scipy.signal import find_peaks
        # print('\n\nmax_coords', img.argmax( axis=0).shape )
        y_coords1, y_coords2, y_coords3 = [], [], []


        gl_max1, gl_max2, gl_min1 = [], [], []
        patch = img[:, :]   

        # print('\n\nPatch_shape',patch.shape[0])
        for i in range(patch.shape[1]): 
            # max1
            a_scan_line = patch[:, i]    
            peaks = find_peaks(a_scan_line, height= 0)
            max1 = np.array(peaks[1]['peak_heights']).argsort()[-1]
            gl_max1.append(peaks[0][max1])

            # min1
            min1 = find_min1(peaks[0][max1], a_scan_line )
            gl_min1.append(min1)

            # max2
            # print('max1',peaks[0][max1])
            peaks1 = find_peaks(patch[ :min1-10, i], height= 0)
            max2 = np.array(peaks1[1]['peak_heights']).argsort()[-1] 
            gl_max2.append(peaks1[0][max2] )

        y_coords1, y_coords3 = associate_y_coords(gl_max1, gl_max2)
        y_coords2 = gl_min1
        # print(y_coords2, '\n')
        # plot_intensity_profile_grid(patch, gl_max1, gl_max2)
        # print('Hi',len(y_coords2), len(y_coords3), len(y_coords1) )

        return y_coords1, y_coords2, y_coords3

    y_coords1, y_coords2, y_coords3 = find2max1min(img)
    # ===========================

    def postprocess_y_coords(y_coords):
        '''Smooth out y_coords depending on avg. of last 10 y_coords
        and some threshold and finally make equal sized array and 
        return'''
        first10_avg = stats.mean( y_coords[5:15])

        for i in range(15, len(y_coords)-5):
            if abs(y_coords[i] - first10_avg) < 1:
                y_coords[i] = y_coords[i]
            else:
                y_coords[i] = (y_coords[i-5] + y_coords[i+5])/2
        return y_coords





    def plot_grid_once_and_for_all(original_gray_img, img, scatter=False, 
                                    y_coords3=[], 
                                    y_coords2=[], 
                                    y_coords1=[],
                                    title='',
                                    plotOn=False,
                                    postprocess=True,
                                    poly_plot=False,
                                    poly_x=[],
                                    poly_y=[]):
        # define fig.
        fig = plt.figure(figsize=(6,6))
        sub_fig_list = [original_gray_img]
        sub_fig_label = [f'Original_Gray_Scaled_{idx}', f'After_k_iters_{idx}']
        # print('y_coords', len(y_coords1))

        if scatter == True:
            print('y_coords1_non-empty\n')    
            y_coords1_scatter, y_coords2_scatter, y_coords3_scatter = [], [], []
            if postprocess == True:
                y_coords1 = postprocess_y_coords(y_coords1)
            y_coords1_scatter = list(y_coords1)
            x_coords1_scatter = list(range(img.shape[1]))
            plt.scatter(x=list(range(len(y_coords1_scatter)))[100:400] , y=y_coords1_scatter[100:400], c='cyan', s=0.1) 
            sub_fig_list.append(fig)


            print('y_coords3_non-empty\n')
            if postprocess == True:
                y_coords3 = postprocess_y_coords(y_coords3)
            y_coords3_scatter = list(y_coords3)
            x_coords3_scatter = list(range(img.shape[1]))
            # plt.scatter(x=list(range(len(y_coords3_scatter)))[100:400] , y=y_coords3_scatter[100:400], c='r', s=0.1)
        
        
            print('y_coords2_non-empty\n')
            if postprocess == True:
                y_coords2 = postprocess_y_coords(y_coords2)
            y_coords2_scatter = list(y_coords2)
            x_coords2_scatter = list(range(img.shape[1]))
            # plt.scatter(x=list(range(len(y_coords2_scatter)))[100:400] , y=y_coords2_scatter[100:400], c='yellow', s=0.1)


            if plotOn == True:
                implot = plt.imshow(img, cmap='gray')
                plt.title(title)

                if poly_plot == True:
                    plt.plot(poly_x, poly_y, color='red')

                plt.show()

                



        else:
            for i in range(len(sub_fig_list)):
                
                fig.add_subplot( 1, 2 , i+1)
                plt.imshow(sub_fig_list[i], cmap='gray')
                plt.title(sub_fig_label[i])
                plt.show()
        

        return y_coords1_scatter, y_coords2_scatter, y_coords3_scatter

    # print("y_coords1_original_gray_img_find2max1min: ",y_coords1[150:400])
    y_coords1_scatter, y_coords2_scatter, y_coords3_scatter = plot_grid_once_and_for_all(  original_gray_img,  img,  scatter=True, 
                                y_coords3=y_coords3, 
                                y_coords2=y_coords2, 
                                y_coords1=y_coords1,
                                title=f"{path.split('/')[-1]}_original_gray_img_y_coords_1",
                                plotOn=True )

    # plot_grid_once_and_for_all(  original_gray_img,  img,  scatter=False, 
    #                             y_coords3=y_coords3, y_coords2=y_coords2, y_coords1=y_coords1 )



    # Find Canny Edge Maps which results in edge-pixels
    def perform_Canny_Edge_Detection_and_plot(gray, title=''):
        
        # print('shape of image', gray.shape)
        # print(np.amax(gray))
        # gray = gray/255.0
        # define lower and upper thresh
        lower = 180
        upper = 240

        edges = cv2.Canny(gray, lower, upper)
        plt.imshow(edges, 'gray')
        plt.title(title)
        # plt.imsave(f'../out_files/{paths[0].split("/")[-1]}_mb{gb}->gb_ce(3,3).png', edges, cmap='gray')
        plt.show()
        return edges
    # blurred_gray_img = read_n_convert2gray( path, gb=True)
    # perform_Canny_Edge_Detection_and_plot( blurred_gray_img)
    # canny_edge_map = perform_Canny_Edge_Detection_and_plot( 
    #                                     original_gray_img, 
    #                                     title=f"{path.split('/')[-1]}_cem")

    # roi_x = list(range(len(y_coords3_scatter)))[100:400]
    # roi_y = y_coords1_scatter[100:400]

    inp_edge_map = img
    sobelY = cv2.Sobel(inp_edge_map, cv2.CV_64F, 0, 1)
    # print('sobelY_max_min', np.amax(sobelY), np.amin(sobelY) )
    
    # Using +ve grads only construct edges
    # print(inp_edge_map.shape, sobelY.shape)
    for j in range(inp_edge_map.shape[0]):
        for i in range(inp_edge_map.shape[1]):
            sobelY[j,i] = max(0, sobelY[j,i])

    

    # print("y_coords1_original_img_find2max1min: ",y_coords1[150:400])
    plt.imshow(sobelY, cmap='gray')
    plt.title(f"{path.split('/')[-1]}_Method1")
    plt.show()


    # y_coords1, y_coords2, y_coords3 = find2max1min( sobelY )
    min_y, max_y = np.argmin(y_coords1_scatter[100:400]), np.argmax(y_coords1_scatter[100:400])
    # print('min-max\n',min_y, max_y)



    y_coords1_sobelY, y_coords2_sobelY, y_coords3_sobelY = find2max1min(sobelY)
    # print('len of coordinates\n', len(y_coords1_sobelY), len(y_coords2_sobelY), len(y_coords3_sobelY))

    # Using sobelY, min_y, max_y ==> plot the +ve grads in sobelY
    # y_coords3_sobel, y_coords2_sobel, y_coords1_sobel = [], [], sobelY[:, min_y : max_y]
    # plot_rnfl_sobelY(sobelY, min_y, max_y)


    # method2_title = f"{path.split('/')[-1]}_rnfl_only_find2max1min(sobelY)"
    method2_title = f"{path.split('/')[-1]}_Method2"
    y_coords1_scatter_sobelY, y_coords2_scatter_sobelY, y_coords3_scatter_sobelY = plot_grid_once_and_for_all(  original_gray_img,  sobelY,  
                                                                            scatter=True, 
                                                                            y_coords3=y_coords3_sobelY,    
                                                                            y_coords2=y_coords2_sobelY, 
                                                                            y_coords1=y_coords1_sobelY, 
                                                                            title = method2_title,
                                                                            plotOn=True )


    top_layer_highest_acc_y_coords = []
    print('\n\n=>> sobel[j,i]\n\n')
    for i in range(sobelY.shape[1]):
        for j in range(320, 600):
            if(sobelY[j,i]*1e15 > 1):
                print(' ', j, sobelY[j,i]*1e15)
                top_layer_highest_acc_y_coords.append(j)
                break #found the upper-most layer
        print(f'\nnew_a_scan for {i}: \n', i)

    y_coords1_best, y_coords2_best, y_coords3_best = top_layer_highest_acc_y_coords, [], []
    # method3_title=f"{path.split('/')[-1]}_best"
    method3_title=f"{path.split('/')[-1]}_Method3"
    y_coords1_scatter_method3, y_coords2_scatter_method3, y_coords3_scatter_method3 = plot_grid_once_and_for_all(  original_gray_img,  sobelY,  
                                                                            scatter=True, 
                                                                            y_coords3=y_coords3_best,    
                                                                            y_coords2=y_coords2_best, 
                                                                            y_coords1=y_coords1_best, 
                                                                            title=method3_title,
                                                                            postprocess=False,
                                                                            plotOn=True
                                                                            )



    # print('sobelY.shape: ',sobelY.shape)
    # plt.imshow(sobelY, cmap='gray')
    # plt.title(f"{path.split('/')[-1]}_sobelY_test")
    # plt.show()

    # Just use Method3 y_coord1 as inp to polyfit and get the poly
    # plot the poly along with the sobelY image w/o 

    x_given = list(range(len(y_coords1_scatter_method3[100:400])))
    print(f'\n{len(y_coords1_scatter_method3[100:400])}, {len(x_given)}\n')
    y_given = y_coords1_scatter_method3[100:400]
    x_poly_domain_lb, x_poly_domain_ub = 0, len(y_given)
    y_poly, p_coeff_list = curve_fitting(x_given, y_given, x_poly_domain_lb, 
                                        x_poly_domain_ub, 
                                        type='poly_fit', plot=False, 
                                        pOrder=10)

    print(y_poly)
    print(len(y_poly))
    print(p_coeff_list)


    x = np.linspace(x_poly_domain_lb, x_poly_domain_ub, 1000) + 100
    y = y_poly
    print(len(x), len(y))

    temp = []
    for i in range(len(x)):
        if i >= 200 and i <= 280:
            temp.append( [x[i], y[i]] )

    a = np.array(temp)
    # print('1st 5 a: \n',a[:5])
    from find_curvature import curvature

    curve_score = curvature(a)
    print("Hi")
    pos = np.argmax(curve_score)
    val = np.max(curve_score)

    print(f'\n curve_score: {curve_score}, pos: {pos}, val: {val}\n')
    print("Hi")

    alt_title=f"{path.split('/')[-1]}_Method3_poly_fit"
    y_coords1_scatter_method3, y_coords2_scatter_method3, y_coords3_scatter_method3 = plot_grid_once_and_for_all(  original_gray_img,  sobelY,  
                                                                            scatter=True, 
                                                                            y_coords3=y_coords3_best,    
                                                                            y_coords2=y_coords2_best, 
                                                                            y_coords1=y_coords1_best, 
                                                                            title=alt_title,
                                                                            postprocess=False,
                                                                            poly_plot=True,
                                                                            poly_x=x,
                                                                            poly_y=y,
                                                                            plotOn=True )


    img_idx = path.split('/')[-1].split('.')[0].split('_')[-1]
    global_dict['img_idx'].append(img_idx)
    global_dict['val'].append(val)

    return global_dict





    


    # ILM-RNFL (CPxs ) --> (+ve ig, above approx RNFL)

    # Select Nodes

    # Compute edge-wts

    # Dijkstra

    # Extrapolate 



if __name__ == '__main__':
    paths = glob.glob('/Users/prateek/Desktop/Segmetation_github/Segmentation-Project/central_scan_task/data_files/OCT_4_sample_folders/2675931_21017_0_0/*.png')
    global_dict = {'img_idx':[], 'val':[]}
    for i in range(128):
        # print(paths[i])
        global_dict = aprxTRL(paths[i], i, global_dict)
        print(f'{i}')

    print(global_dict)
    fp = open('./global_dict_2675931_21017_0_0.json', 'w')
    fp.write(json.dumps(global_dict))
    