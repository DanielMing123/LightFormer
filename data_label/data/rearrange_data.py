import os 
import json
import random
from random import shuffle

def gather_info(img_folder):

    interval = 1
    # st_green = [[24750, 25044], [25934, 26564], [26884, 27054], [27476, 27644], [27970, 28176], [28468, 28748],  
    # [29046, 29318], [29604, 29804], [30608, 30754], [31718, 31862], [32890, 33042], [34306, 34422], [35036, 35434], [36084, 36690],[37832, 38022], [38352, 38428], [39194, 39400], [40066, 40344]]
    # st_red = [[25288, 25930], [28984, 29042], [33048, 33910], [34760, 35030], [35858, 36080], [37068, 37828], [38628, 39190], [39542, 40060]]
    st_lf_green = [[120,190],[350,468],[667,723],[1311,1453]]
    st_lf_red = [[520,665]]
    st_g_lf_r = [[0,119]]
    st_r_lf_g = []

    all_imgs = sorted(os.listdir(img_folder))
    final_res_st_lf_green = []
    final_res_st_lf_red = []
    final_res_st_g_lf_r = []
    final_res_st_r_lf_g = []
    # 专门处理直行左转绿灯
    for start_frame, end_frame in st_lf_green:
        # start_idx = all_imgs.index(str(start_frame) + '.png')
        # end_idx = all_imgs.index(str(end_frame) + '.png')
        start_idx = start_frame
        end_idx = end_frame
        count = start_idx
        img_list = []
        # import pdb;pdb.set_trace()
        while count <= end_idx:
            img_list.append(all_imgs[count])
            count = count + interval
            if len(img_list) == 10:
                final_res_st_lf_green.append(dict(images = img_list, label = [1, 0, 1, 0]))
                img_list = []
    
    #专门处理直行左转红灯
    for start_frame, end_frame in st_lf_red:
        # start_idx = all_imgs.index(str(start_frame) + '.png')
        # end_idx = all_imgs.index(str(end_frame) + '.png')
        start_idx = start_frame
        end_idx = end_frame
        count = start_idx
        img_list = []
        while count <= end_idx:
            img_list.append(all_imgs[count])
            count = count + interval
            if len(img_list) == 10:
                final_res_st_lf_red.append(dict(images = img_list, label = [0, 1, 0, 1]))
                img_list = []

    #专门处理直行绿灯左转红灯
    for start_frame, end_frame in st_g_lf_r:
        # start_idx = all_imgs.index(str(start_frame) + '.png')
        # end_idx = all_imgs.index(str(end_frame) + '.png')
        start_idx = start_frame
        end_idx = end_frame
        count = start_idx
        img_list = []
        while count <= end_idx:
            img_list.append(all_imgs[count])
            count = count + interval
            if len(img_list) == 10:
                final_res_st_g_lf_r.append(dict(images = img_list, label = [1, 0, 0, 1]))
                img_list = []

    #专门处理直行红灯左转绿灯
    for start_frame, end_frame in st_r_lf_g:
        # start_idx = all_imgs.index(str(start_frame) + '.png')
        # end_idx = all_imgs.index(str(end_frame) + '.png')
        start_idx = start_frame
        end_idx = end_frame
        count = start_idx
        img_list = []
        while count <= end_idx:
            img_list.append(all_imgs[count])
            count = count + interval
            if len(img_list) == 10:
                final_res_st_r_lf_g.append(dict(images = img_list, label = [0, 1, 1, 0]))
                img_list = []

    final_res = final_res_st_lf_green + final_res_st_lf_red + final_res_st_g_lf_r + final_res_st_r_lf_g
    shuffle(final_res)
    return final_res

def write_json(json_res, out_path):
    with open(os.path.join(out_path, 'samples.json'), 'w') as f:
        json.dump(json_res, f)

def write_train_test_json(json_res, green_res, red_res, out_path=None):
    all_green_samples = len(green_res)
    all_red_samples = len(red_res)
    train_res = []
    test_res = []
    train_green_idxs = random.sample(range(0,all_green_samples), 50)
    train_red_idxs = random.sample(range(0,all_red_samples), 50)
    for train_gr_idx in train_green_idxs:
        train_res.append(green_res[train_gr_idx])
    for train_rd_idx in train_red_idxs:
        train_res.append(red_res[train_rd_idx])
    shuffle(train_res)

    for content in json_res:
        if content not in train_res:
            test_res.append(content)
    shuffle(test_res)

    print(all_green_samples, all_red_samples)
    print(len(train_res), len(test_res))

    with open(os.path.join(out_path, 'train/samples2.json'), 'w') as f:
        json.dump(train_res, f) 

    with open(os.path.join(out_path, 'test/samples2.json'), 'w') as f:
        json.dump(test_res, f) 

def rearrange_json(in_path, out_path=None):
    samples = None
    new_output = []
    with open(in_path,'r') as f:
        samples = json.load(f)
    
    # generate new labels
    for sample in samples:
        if sample["label"][0] == 1:
            sample["label"][2] = 1
        elif sample["label"][1] == 1:
            sample["label"][3] = 1
        new_output.append(sample)
    
    # write new labels into json file
    with open(os.path.join(out_path, 'samples.json'), 'w') as f:
        json.dump(new_output, f) 

if __name__ == "__main__":
    img_folder = '/media/tao/Data_Use/Light_Former/data/Kaggle_Dataset/nightTrain/nightTrain/nightClip5/frames'
    out_path = '/media/tao/Data_Use/Light_Former/data/Kaggle_Dataset/nightTrain/nightTrain/nightClip5/train'
    json_res = gather_info(img_folder)
    write_json(json_res, out_path)
    # write_train_test_json(json_res, green_res, red_res, out_path)
    # in_path = "/media/tao/Data_Use/Traffic_Light_Model/prediction_ml_framework/data/balanced_data/train/samples.json"
    # out_path = "/media/tao/Data_Use/Traffic_Light_Model/prediction_ml_framework/data/balanced_data/st_lf_test"
    # rearrange_json(in_path, out_path)