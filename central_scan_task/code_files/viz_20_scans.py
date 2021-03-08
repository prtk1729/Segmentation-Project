import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

paths = glob.glob('/Users/prateek/Desktop/Segmetation_github/Segmentation-Project/central_scan_task/data_files/OCT_4_sample_folders/2675931_21017_0_0/*.png')
    
def plot_scans():    

    fig = plt.figure(figsize = (15,12))
    title_list = [f'{str(i)}th_scan' for i in range(15)]
    # display = 

    for i in range(15):

        fig.add_subplot(3, 5, i+1)
        img = mpimg.imread(paths[i])
        plt.imshow(img, cmap='gray')
        plt.title(title_list[i])

    plt.show()

plot_scans()
