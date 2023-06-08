import os 
import numpy as np

def seperate_precision_recall_f1_analysis(txt_path):
    st_green_tp = 0
    st_grean_tp_fp = 0
    st_green_tp_fn = 0
    st_red_tp = 0
    st_red_tp_fp = 0
    st_red_tp_fn = 0
    st_green_precision = 0
    st_green_recall = 0
    st_green_f1 = 0
    st_red_precision = 0
    st_red_recall = 0
    st_red_f1 = 0

    lf_green_tp = 0
    lf_grean_tp_fp = 0
    lf_green_tp_fn = 0
    lf_red_tp = 0
    lf_red_tp_fp = 0
    lf_red_tp_fn = 0
    lf_green_precision = 0
    lf_green_recall = 0
    lf_green_f1 = 0
    lf_red_precision = 0
    lf_red_recall = 0
    lf_red_f1 = 0

    with open(txt_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            info = line.split(' ')
            # import pdb;pdb.set_trace()
            # process straight tp+fp
            if int(info[-5]) == 1:
                st_grean_tp_fp += 1
                if info[-5] == info[-4]:
                    st_green_tp += 1
            else:
                st_red_tp_fp += 1
                if info[-5] == info[-4]:
                    st_red_tp += 1
            # process left turn tp+fp
            if int(info[-3]) == 1:
                lf_grean_tp_fp += 1
                if info[-3] == info[-2]:
                    lf_green_tp += 1
            else:
                lf_red_tp_fp += 1
                if info[-3] == info[-2]:
                    lf_red_tp += 1
            # process straight tp+fn
            if int(info[-4]) == 1:
                st_green_tp_fn += 1
            else:
                st_red_tp_fn += 1
            # process left turn tp+fn
            if int(info[-2]) == 1:
                lf_green_tp_fn += 1
            else:
                lf_red_tp_fn += 1

    st_green_precision = st_green_tp / st_grean_tp_fp
    st_green_recall = st_green_tp / st_green_tp_fn
    st_green_f1 = 2*st_green_precision * st_green_recall / (st_green_precision + st_green_recall)
    
    st_red_precision = st_red_tp / st_red_tp_fp
    st_red_recall = st_red_tp / st_red_tp_fn
    st_red_f1 = 2 * st_red_precision * st_red_recall / (st_red_precision + st_red_recall)

    lf_green_precision = lf_green_tp / lf_grean_tp_fp
    lf_green_recall = lf_green_tp / lf_green_tp_fn
    lf_green_f1 = 2* lf_green_precision * lf_green_recall / (lf_green_precision + lf_green_recall)

    lf_red_precision = lf_red_tp / lf_red_tp_fp
    lf_red_recall = lf_red_tp / lf_red_tp_fn
    lf_red_f1 = 2 * lf_red_precision * lf_red_recall / (lf_red_precision + lf_red_recall)

    print(f"Go Straight Pass precision:{st_green_precision*100:.2f}%, recall:{st_green_recall*100:.2f}%, F1 score:{st_green_f1*100:.2f}%")
    print(f"Go Straight Stop precision:{st_red_precision*100:.2f}%, recall:{st_red_recall*100:.2f}%, F1 score:{st_red_f1*100:.2f}%")
    print(f"Left Turn Pass precision:{lf_green_precision*100:.2f}%, recall:{lf_green_recall*100:.2f}%, F1 score:{lf_green_f1*100:.2f}%")
    print(f"Left Turn Stop precision:{lf_red_precision*100:.2f}%, recall:{lf_red_recall*100:.2f}%, F1 score:{lf_red_f1*100:.2f}%")

def seperate_analysis(txt_path):
    data_num = 0
    st_correct_num = 0
    lf_correct_num = 0
    general_correct_num = 0
    with open(txt_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            info = line.split(' ')
            data_num += 1
            if info[-2] == info[-3]:
                lf_correct_num += 1
            if info[-4] == info[-5]:
                st_correct_num += 1
            if info[-1] == 'right':
                general_correct_num += 1
    # return data_num, correct_num
    st_acc = st_correct_num/data_num * 100
    lf_acc = lf_correct_num/data_num * 100
    gen_acc = general_correct_num/data_num * 100
    print(f'Go Straight accuracy:{st_acc:.2f}%')
    print(f'Left Turn accuracy:{lf_acc:.2f}%')
    print(f'Overall accuracy:{gen_acc:.2f}%')

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
    txt_path = './complete_model_Kaggle_night_n=20_res1.txt'
    # analysis(txt_path=txt_path)
    seperate_analysis(txt_path)
    seperate_precision_recall_f1_analysis(txt_path)
    # folder_path = '/media/tao/Data_Use/Traffic_Light_Model/prediction_ml_framework/test_res'
    # main(folder_path)






