import json
import numpy as np

fp = open('/Users/prateek/Desktop/Segmetation_github/Segmentation-Project/central_scan_task/code_files/global_dict_2675931_21017_0_0.json', 'r')
d = json.load(fp)

idx = np.argmax(np.array(d['val']))
img_idx = d['img_idx'][idx]
print(f'Index of image with the maximum curvature: {img_idx}')