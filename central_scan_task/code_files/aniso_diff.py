import numpy as np
import warnings
import cv2
import glob
import matplotlib.pyplot as plt

def anisodiff(img,niter=1,kappa=32,gamma=0.1,step=(1.,1.),option=1,ploton=False):
    """
    Anisotropic diffusion.
    """
 
    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)
 
    # initialize output array
    img = img.astype('float64')
    imgout = img.copy()
 
    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()
 
    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep
 
        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
 
        ax1.imshow(img,interpolation='nearest')
        ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")
 
        fig.canvas.draw()
 
    for ii in range(niter):
 
        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)
 
        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
 
        # update matrices
        E = gE*deltaE
        S = gS*deltaS
 
        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]
 
        # update the image
        temp_img = gamma*(NS+EW)
        imgout += gamma*(NS+EW)
 
        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            print('Line_127')
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)
 
    return imgout

def read_n_convert2gray(path, gb=True):

    global paths
    # read the img
    img = cv2.imread(path)
    print(img.shape)

    # copy the img (recall best practice)
    img_copy = np.copy(img)

    # ColorSpace tranform
    img = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
    return img

def aniso_driver(gray_img, plot=True):
    # paths = glob.glob('/Users/prateek/Desktop/Segmetation_github/Segmentation-Project/central_scan_task/data_files/OCT_4_sample_folders/2675931_21017_0_0/*.png')

    for i in range(1):
        # gray_img = read_n_convert2gray(paths[i], gb=False)

        if plot == True:
            plt.imshow(gray_img, cmap='gray')
            plt.show()

        print(gray_img.shape)
        imgout = anisodiff(gray_img, niter = 1, ploton=False)

        if plot == True:
            plt.imshow(gray_img, cmap='gray')
            plt.show()

    return imgout