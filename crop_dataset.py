#-*- coding: utf-8 -*-
import json
import os
import cv2
import argparse


def crop(name,img_path,label_path,save_path,start,end,train=True):

    """
    Crop Images with bounding box in annotations
    ARGS:
        img_path : input folder path where starts imagePath
        label_path : list of label path
        save_path     : list of image path
        start : start of cropped img number for each folder
        end : end of cropped img number for each folder
        train : if true, Load/Save Image in Train , if False, Load/Save Image in Validation
    """
    os.makedirs(save_path + "train/",exist_ok=True)
    os.makedirs(save_path + "validation/",exist_ok=True)
    gt_train_file = open(save_path + 'gt_train.txt', 'a')
    gt_valid_file = open(save_path + 'gt_valid.txt', 'a')

    for i in range(start,end):
        try:
            annotations = json.load(open(label_path+name+'_'+ (str(i).zfill(6)) + '.json'))

            image = cv2.imread(img_path + name+ '_'+(str(i).zfill(6)) + '.jpg')
            if (image is None):
                image = cv2.imread(img_path + name+ '_'+(str(i).zfill(6)) + '.JPG')
            if (image is None):
                image = cv2.imread(img_path + name+ '_'+(str(i).zfill(6)) + '.JPEG')
            if (image is None):
                continue
            for idx, annotation in enumerate(annotations['annotations']):
                x, y, w, h = annotations['annotations'][idx]['bbox']

                text = annotations['annotations'][idx]['text']

                if "X" in text: continue
                if "x" in text: continue
                if x < 0: continue
                if y < 0: continue
                crop_img = image[y:y + h, x:x + w]
                crop_file_name = name+ '_'+ (str(i).zfill(6)) + '_{:02}.jpg'.format(idx + 1)
                print(crop_file_name)
                if train == True:
                    cv2.imwrite(save_path + "train/" + crop_file_name, crop_img)
                    gt_train_file.write("train/{}\t{}\n".format(crop_file_name, text))
                else:
                    cv2.imwrite(save_path + "validation/" + crop_file_name, crop_img)
                    gt_valid_file.write("validation/{}\t{}\n".format(crop_file_name, text))
        except:
            print('file not found')
    gt_train_file.close()
    gt_valid_file.close()


def data_preprocessing(opt, train=True):
    """
    Make Image Path - Bounding box lists for cropping
     ARGS:
         opt has
         datset_path : Original dataset path ( AI HUB )
         save_path : save_path to save cropped image and gt file
         train : Decide Target Folder name is Training or Validation
     """
    root = opt.dataset_path
    save_path = opt.save_path
    if(train== True):
        root = root +'Training'
    else:
        root = root + 'Validation'
    folderlist = os.listdir(root + '/image/')
    folderlist.sort()
    print(folderlist)

    ser = ''
    ser_n = 0
    ser_b = ''
    ser_b_n = 0
    for i, folder in enumerate(folderlist):

        img_list  = os.listdir(root+'/image/' + folder)
        img_list.sort()
        img_path = root + '/image/'+folder +'/'
        tmp_s = img_list[0].split('_')
        if (img_list[0][-1] == 'p'):
            tmp_s = img_list[1].split('_')
        if(tmp_s[0] == '간판'):

            start = int(tmp_s[2][:6])
            tmp_s = img_list[-1].split('_')
            end = int(tmp_s[2][:6])
            name = tmp_s[0] + '_' + tmp_s[1]
            label_root = root + '/label/1.간판/'

            if(tmp_s[1] != ser):
                ser = tmp_s[1]
                ser_n = ser_n + 1
                label_path = label_root +str(ser_n) +'.' + tmp_s[1]+'/'

                if( train ==True):
                    label_path = label_root +str(ser_n) +'.' + tmp_s[1]+'/' + tmp_s[1]+ folder[-1] +'/'
            else:
                label_path = label_root + str(ser_n) + '.' + tmp_s[1] + '/'

                if( train ==True):
                    label_path = label_root + str(ser_n) + '.' + tmp_s[1] + '/' + tmp_s[1] + folder[-1] + '/'
        else:
            start = int(tmp_s[2][:6])
            tmp_s = img_list[-1].split('_')
            end = int(tmp_s[2][:6])
            name = tmp_s[0] + '_' + tmp_s[1]
            label_root = root + '/label/2.책표지/'

            if(tmp_s[1] != ser_b):
                ser_b = tmp_s[1]
                ser_b_n = ser_b_n + 1
                label_path = label_root +str(ser_b_n).zfill(2) +'.' + tmp_s[1]+'/'
                print(label_path)
        if(train == True):
            crop(name,img_path,label_path,save_path,start,end,True)
        else:
            crop(name,img_path,label_path,save_path,start,end,False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="/data/야외 실제 촬영 한글 이미지/", help='image and annotations path')
    parser.add_argument('--save_path', default="data/",
                        help='save_path to save cropped image and gt file')
    opt = parser.parse_args()
    data_preprocessing(opt, True)
    data_preprocessing(opt, False)