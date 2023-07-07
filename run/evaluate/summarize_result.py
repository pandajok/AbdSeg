
import os
import sys
import numpy as np
import pandas as pd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from utils.file_utils import load_df



def get_1fold_average_result(result_path):
    file_name = 'ind_seg_result.csv'
    result_path += file_name
    data_df = load_df(result_path)
    column_header = ['1_DSC', '1_NSC', '2_DSC', '2_NSC', '3_DSC', '3_NSC', '4_DSC', '4_NSC',
                     '5_DSC', '5_NSC', '6_DSC', '6_NSC', '7_DSC', '7_NSC', '8_DSC', '8_NSC',
                     'Average_DSC', 'Average_NSC', 'Data_loader_time',
                     'Coarse_infer_time', 'Coarse_postprocess_time', 'Fine_infer_time', 'Time_usage',
                     'Memory_usage', 'Time_score', 'Memory_score']
    df_column_header = data_df.columns.values
    for name in column_header:
        if name in df_column_header:
            data = data_df[name].values
            print('{}: {}'.format(name, np.mean(data)))


def get_5fold_average_result(result_dir):
    result_folder = os.listdir(result_dir)
    all_result = []
    column_header = ['1_DSC', '1_NSC', '2_DSC', '2_NSC', '3_DSC', '3_NSC', '4_DSC', '4_NSC',
                     '5_DSC', '5_NSC', '6_DSC', '6_NSC', '7_DSC', '7_NSC', '8_DSC', '8_NSC', 'Average_DSC', 'Average_NSC']
    for fold in result_folder:
        result_path = result_dir + fold + '/ind_seg_result.csv'
        data_df = load_df(result_path)
        df_column_header = data_df.columns.values
        fold_result = []
        print('\n{}:'.format(fold))
        for name in column_header:
            if name in df_column_header:
                data = data_df[name].values
                fold_result.append(data)
                print('mean {}: {}'.format(name, np.mean(data)))
                print('std {}: {}'.format(name, np.std(data)))
        all_result.append(fold_result)
    all_result = np.array(all_result)
    print('\naverage of 5-fold cross validation: ')
    for idx, name in enumerate(column_header):
        print('mean {}: {}'.format(name, np.mean(all_result[:, idx])))
        print('std {}: {}'.format(name, np.std(all_result[:, idx])))


if __name__ == '__main__':
    result_path = './output/results/'
    get_1fold_average_result(result_path)
#     result_dir = './output/results/'
#     get_5fold_average_result(result_dir)
