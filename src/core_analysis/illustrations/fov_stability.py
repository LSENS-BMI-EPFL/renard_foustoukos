import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(r'/home/aprenard/repos/NWB_analysis')
sys.path.append(r'/home/aprenard/repos/fast-learning')
import src.utils.utils_io as io
from nwb_wrappers.nwb_reader_functions import get_image_mask


ops_paths = [
    '/mnt/lsens-analysis/Anthony_Renard/data/AR180/AR180_20241213_150420/suite2p/plane0/ops.npy',
    '/mnt/lsens-analysis/Anthony_Renard/data/AR180/AR180_20241214_194639/suite2p/plane0/ops.npy',
    '/mnt/lsens-analysis/Anthony_Renard/data/AR180/AR180_20241215_190049/suite2p/plane0/ops.npy',
    '/mnt/lsens-analysis/Anthony_Renard/data/AR180/AR180_20241216_145407/suite2p/plane0/ops.npy',
    '/mnt/lsens-analysis/Anthony_Renard/data/AR180/AR180_20241217_160355/suite2p/plane0/ops.npy',
]

mean_images = []
for path in ops_paths:
    ops = np.load(io.adjust_path_to_host(path), allow_pickle=True)
    mean_images.append(ops.item()['meanImg'])

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, (ax, mean_img) in enumerate(zip(axes, mean_images)):
    ax.imshow(mean_img, cmap='gray')
    ax.set_title(f'Day {i+1}')
    ax.axis('off')
plt.tight_layout()

plt.savefig('/mnt/lsens-analysis/Anthony_Renard/analysis_output/fast-learning/illustrations/fov_stability.svg', dpi=300, bbox_inches='tight')
