import os 
import numpy as np


def analysis(txt_path, data_num=0, correct_num=0):
    data_num = 0
    correct_num = 0
    with open(txt_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            info = line.split(' ')
            data_num += 1
            if info[-1] == 'right':
                correct_num += 1
    # return data_num, correct_num
    acc = correct_num/data_num * 100
    print(f'accuracy:{acc:.2f}%')

def main(folder_path):
    data_num = 0
    correct_num = 0
    contents = sorted(os.listdir(folder_path))
    for content in contents:
        txt_path = os.path.join(folder_path, content)
        data_num, correct_num = analysis(txt_path, data_num, correct_num)
    acc = correct_num/data_num * 100
    print(correct_num)
    print(data_num)
    print(f'accuracy:{acc:.2f}%')



if __name__ == '__main__':
    txt_path = './MZX_former_Bosch_res.txt'
    analysis(txt_path=txt_path)
    # folder_path = '/media/tao/Data_Use/Traffic_Light_Model/prediction_ml_framework/test_res'
    # main(folder_path)



