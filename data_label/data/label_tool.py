import os 
import json
import cv2
import argparse

def label_sample(label, img):
    for i in range(2):
        cv2.imshow('tlj',img)
        key = cv2.waitKey(0)
        # print(key)
        if key == 49:
            label['label'][0] = 1
        elif key == 50:
            label['label'][1] = 1
        elif key == 51:
            label['label'][2] = 1
        elif key == 52:
            label['label'][3] = 1
        elif key == 13:
            return None
        elif key == 119:
            return -1
        elif key == 115:
            return 1
    return label

def label_tool(json_path, st_idx=0):
    labels = None
    with open(json_path,'r') as f:
        labels = json.load(f)
    total = len(labels)
    idx = st_idx
    while True:
        label = labels[idx]
        if sum(label['label']) <= 1:
            print('需要标注')
        img_names = label['images']
        label['label'] = [0,0,0,0]
        img_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(json_path)), 'frames'), img_names[2]) # .split('_120.jpg')[0] + '_30.jpg'
        print(f'{idx+1}/{total}')
        print(img_path)
        img = cv2.imread(img_path)
        label = label_sample(label, img)
        
        #用于保存阶段性标注成果以及往前面或者后面的sample移动
        if (label == None) or idx == (total-1):
            with open(os.path.dirname(json_path) + '/samples.json','w') as f:
                json.dump(labels, f)
            print(f'退出程序, 目前idx:{idx}')
            return
        elif label == 1:
            idx += 1
            continue
        elif label == -1:
            idx -= 1
            continue
        print(label['label'])

        # 防止标注出现错误
        while sum(label['label'][:2]) == 2 or sum(label['label'][2:]) == 2:
            label['label'] = [0,0,0,0]
            label = label_sample(label, img)
            print(label['label'])
        
        idx += 1

def label_all(upper_level):
    all_folders = os.listdir(upper_level)
    for folder in all_folders:
        folder_path = os.path.join(upper_level, folder)
        all_subfolds = os.listdir(folder_path)
        for subfold in all_subfolds:
            if '.json' in subfold:
                continue
            content = os.listdir(os.path.join(folder_path, subfold, 'train'))
            if 'new_samples.json' in content:
                continue
            json_path = os.path.join(folder_path, subfold, 'train','samples.json')
            print(json_path)
            label_tool(json_path,sample_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Jump to certain folder and file.')
    parser.add_argument('--index',type=int, default=0,help="Target index")
    args = parser.parse_args()
    sample_idx = args.index
    # upper_level = '/media/tao/Data_Use/Traffic_Light_Model/prediction_ml_framework/data/shunyi_cityzone_train'
    # label_all(upper_level)
    json_path = "/media/tao/Data_Use/Light_Former/data/Bosch_dataset/test/samples_1.json"
    label_tool(json_path, sample_idx)





    
