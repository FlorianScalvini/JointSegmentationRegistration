import torch
import torchio as tio
import glob

path_data = glob.glob("/home/florian/Documents/Dataset/babofet/babofetFiloutte/derivatives/longiseg/sub-*/*/anat/*.nii.gz")

for path in path_data:
    seg = tio.LabelMap(path)
    print('Number of unique labels in segmentation:', torch.unique(seg.data))

