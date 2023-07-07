
import os
import sys
import numpy as np
from tqdm import tqdm
from multiprocessing import Process

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from utils.image_io import load_ct_info
from utils.file_utils import read_txt, write_csv
from utils.metric import compute_flare_metric1,compute_hausdorff_distance


def process_label(label):
    spleen = label == 1
    right_kidney = label == 2
    left_kidney = label == 3
    gallbladder = label == 4
    # esophagus = label == 5
    liver = label == 6
    stomach = label == 7
    aorta = label == 8
    # inferior_vena_cava = label == 9
    # portal_vein_splenic_vein = label == 10
    pancreas = label == 5
    # right_adrenal_gland = label == 12
    # left_adrenal_gland = label == 13

    return spleen, right_kidney, left_kidney, gallbladder, pancreas, liver, stomach, aorta


def process_compute_metric(name, gt_dir, predict_dir, csv_path):
    print('process {} start...'.format(name))
    gt_mask_path = gt_dir + name + '.nii.gz'
    predict_mask_path = predict_dir + name + '.nii.gz'
    num_class = 8

    gt_dict = load_ct_info(gt_mask_path)
    predict_dict = load_ct_info(predict_mask_path)
    gt_mask = gt_dict['npy_image']
    predict_mask = predict_dict['npy_image']
    spacing = gt_dict['spacing']

    mask_shape = gt_mask.shape
    gt_mask_czyx = np.zeros([num_class, mask_shape[0], mask_shape[1], mask_shape[2]])
    predict_mask_czyx = np.zeros([num_class, mask_shape[0], mask_shape[1], mask_shape[2]])
    for i in range(8):
        if i==4:
            gt_mask_czyx[i] = gt_mask == i + 2
            predict_mask_czyx[i] = predict_mask == i + 2
        elif i==5:
            gt_mask_czyx[i] = gt_mask == i + 2
            predict_mask_czyx[i] = predict_mask == i + 2
        elif i==6:
            gt_mask_czyx[i] = gt_mask == i + 2
            predict_mask_czyx[i] = predict_mask == i + 2
        elif i==7:
            gt_mask_czyx[i] = gt_mask == i + 4
            predict_mask_czyx[i] = predict_mask == i + 4
        else:
            gt_mask_czyx[i] = gt_mask == i+1
            predict_mask_czyx[i] = predict_mask == i + 1
    area_dice, surface_dice = compute_flare_metric1(gt_mask_czyx, predict_mask_czyx, spacing)
    out_content = [name, spacing[0]]
    total_area_dice = 0
    total_surface_dice = 0
    object_labels = ['1', '2', '3', '4', '5', '6', '7', '8']
    for i in range(num_class):
        out_content.append(area_dice[i])
        out_content.append(surface_dice[i])
        total_area_dice += area_dice[i]
        total_surface_dice += surface_dice[i]
        print('{} DSC: {}, NSC: {}'.format(object_labels[i], area_dice[i], surface_dice[i]))
    out_content.extend([total_area_dice / num_class, total_surface_dice / num_class])
    write_csv(csv_path, out_content, mul=False, mod='a+')
    print('Average_DSC: {}, Average_NSC: {}'.format(total_area_dice / num_class, total_surface_dice / num_class))
    print('process {} finish!'.format(name))


if __name__ == '__main__':
    series_uid_path = './dataset/file_list/val_series_uids.txt'
    gt_mask_dir = './dataset/val_mask/'
    predict_mask_dir = './output/results/'
    out_ind_csv_dir = './output/results/'
    if not os.path.exists(out_ind_csv_dir):
        os.makedirs(out_ind_csv_dir)
    out_ind_csv_path = out_ind_csv_dir + 'ind_seg_result.csv'

    ind_content = ['series_uid', 'z_spacing']
    labels = ['1', '2', '3', '4', '5', '6', '7', '8']
    object_metric = [] 
    for object_name in labels:
        object_metric.extend([object_name + '_DSC', object_name + '_NSC'])
    ind_content.extend(object_metric)
    ind_content.extend(['Average_DSC', 'Average_NSC'])
    write_csv(out_ind_csv_path, ind_content, mul=False, mod='w')

    file_names = read_txt(series_uid_path)

    for file_name in tqdm(file_names):
        process_compute_metric(file_name, gt_mask_dir, predict_mask_dir, out_ind_csv_path)
