import contextlib
import io
import json
import logging
import os.path
import random
import re
import time

import pandas as pd

from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX

from data.process_text import load_and_transform_text
from data.process_touch import load_and_transform_touch, get_touch_transform
from data.process_vision import load_and_transform_vision, get_vision_transform



import argparse
from os.path import join as opj
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import torch
from PIL import Image
from torchvision import transforms
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


SSVTP_dir = '/data/develop/smallz/TLV-Core/tactile_datasets/train/ssvtp/'
TAG_dir = '/data/develop/smallz/TLV-Core/tactile_datasets/train/TAG/dataset/'
obreal_dir = '/data/develop/smallz/TLV-Core/tactile_datasets/objectfolder/real/tactile/'
visgel_dir = '/data/develop/smallz/TLV-Core/tactile_datasets/visgel/images/touch/'
yuan18_dir = '/data/develop/smallz/TLV-Core/tactile_datasets/yuan18/Data_ICRA18/Data/'
TVL_dir = '/data/develop/smallz/TLV-Core/tactile_datasets/TVL/tvl_dataset/hct/'
ycb_dir = '/data/develop/smallz/TLV-Core/tactile_datasets/YCB-Slide/real/'
octopi_dir = '/data/develop/smallz/TLV-Core/tactile_datasets/train/octopi/'
text_dir = '/data/develop/smallz/TLV-Core/tactile_datasets/text/'

TAG_file = '/data/develop/smallz/TLV-Core/tactile_datasets/contact_text_tag_notest.csv'
obreal_file = '/data/develop/smallz/TLV-Core/tactile_datasets/contact_text_obj.csv'
visgel_file = '/data/develop/smallz/TLV-Core/tactile_datasets/contact_visgel.csv'
yuan18_file = '/data/develop/smallz/TLV-Core/tactile_datasets/contact_yuan.csv'
octopi_file = '/data/develop/smallz/TLV-Core/tactile_datasets/train/datasets--xxuan01--TacQuad/snapshots/5f7548b8beb3ae1da53ce3aa9aeb8a98b2dfa454/contact_text_octopi.csv'
TVL_file = '/data/develop/smallz/TLV-Core/tactile_datasets/contact_text_tvl.csv'

tacquad_indoor_dir = '/data/develop/smallz/TLV-Core/tactile_datasets/train/datasets--xxuan01--TacQuad/snapshots/5f7548b8beb3ae1da53ce3aa9aeb8a98b2dfa454/data_indoor/'
tacquad_outdoor_dir = '/data/develop/smallz/TLV-Core/tactile_datasets/train/datasets--xxuan01--TacQuad/snapshots/5f7548b8beb3ae1da53ce3aa9aeb8a98b2dfa454/data_outdoor/'

tacquad_indoor_file = '/data/develop/smallz/TLV-Core/tactile_datasets/train/datasets--xxuan01--TacQuad/snapshots/5f7548b8beb3ae1da53ce3aa9aeb8a98b2dfa454/contact_indoor.csv'
tacquad_outdoor_file = '/data/develop/smallz/TLV-Core/tactile_datasets/train/datasets--xxuan01--TacQuad/snapshots/5f7548b8beb3ae1da53ce3aa9aeb8a98b2dfa454/contact_outdoor.csv'

tacquad_text_dir = '/data/develop/smallz/TLV-Core/tactile_datasets/train/datasets--xxuan01--TacQuad/snapshots/5f7548b8beb3ae1da53ce3aa9aeb8a98b2dfa454/'


class touch_dataset(Dataset):
    def __init__(self, args):
        super().__init__()
        
        self.touch_list = []
        self.vision_list = []
        self.text_list = []
        self.sensor_list = []
        # gelsight 0
        # digit 1
        # gelslim 2
        # gelsight mini 3
        # duragel 4

        # with open(TAG_file,'r') as file:
        #     csv_reader = csv.reader(file)
        #     for row in csv_reader:
        #         folder = row[0]
        #         image_id = row[1]
        #         test_flag = int(row[3])

        #         # A simple resampling method to create more text-vision-touch triplets for GelSight sensor
        #         for tt in range(2):
        #             if test_flag == 1:
        #                 self.text_list.append(-1)
        #             else:
        #                 self.text_list.append(text_dir + 'tag_' + row[2] +'.pt')
        #             self.vision_list.append(TAG_dir + folder + '/video_frame/' + image_id)
        #             self.touch_list.append(TAG_dir + folder + '/gelsight_frame/' + image_id)
        #             self.sensor_list.append(0)



        # for item in os.listdir(SSVTP_dir+'/images_tac/'):
        #     image_id = item.split('_')[1]
        #     tactile_path = SSVTP_dir+'/images_tac/'+item
        #     image_path = SSVTP_dir+'/images_rgb/'+item.replace('tac', 'rgb')
        #     self.text_list.append(text_dir + 'ssvtp_' + image_id +'.pt')
        #     self.touch_list.append(tactile_path)
        #     self.vision_list.append(image_path)
        #     self.sensor_list.append(1)


        # with open(TVL_file,'r') as file:
        #     csv_reader = csv.reader(file)
        #     for row in csv_reader:
        #         image_id = row[0]
        #         self.text_list.append(text_dir + 'tvl_' + row[1] +'.pt')
        #         self.vision_list.append(TVL_dir + image_id.replace('tactile', 'vision'))
        #         self.touch_list.append(TVL_dir + image_id)
        #         self.sensor_list.append(1)


        # with open(octopi_file,'r') as file:
        #     csv_reader = csv.reader(file)
        #     for row in csv_reader:
        #         # A simple resampling method to create more samples for GelSight Mini sensor
        #         for tt in range(3):
        #             self.touch_list.append(octopi_dir+row[0])
        #             self.text_list.append(text_dir+'octopi_'+row[1]+'.pt')
        #             # self.vision_list.append(-1)
        #             self.vision_list.append(octopi_dir+row[0])
        #             self.sensor_list.append(3)



        
        with open(tacquad_indoor_file,'r') as file:
            csv_reader = csv.reader(file)
            now_id = 0
            for row in csv_reader:
                item_name = row[0]

                gelsight_start = int(row[1])
                gelsight_end = int(row[2])

                digit_start = int(row[3])
                digit_end = int(row[4])

                duragel_start = int(row[5])
                duragel_end = int(row[6])

                for t in range(gelsight_start, gelsight_end+1):

                    png_path = tacquad_indoor_dir + item_name +'/gelsight/' + str(t) +'.png'

                    len_touch = len(os.listdir(tacquad_indoor_dir + item_name +'/gelsight/'))
                    len_image = len(os.listdir(tacquad_indoor_dir + item_name +'/img_gelsight/'))
                    time_point = t / (len_touch*1.0)
                    vision_id = int(time_point * len_image)
                    
                    image_path = tacquad_indoor_dir + item_name +'/img_gelsight/' + str(vision_id) +'.png'

                    self.vision_list.append(image_path)
                    self.touch_list.append(png_path)
                    self.text_list.append(tacquad_text_dir + 'tacquad_indoor_' + str(now_id) +'.pt')
                    self.sensor_list.append(3)
                
                for t in range(digit_start, digit_end+1):

                    png_path = tacquad_indoor_dir + item_name +'/digit/' + str(t) +'.png'

                    len_touch = len(os.listdir(tacquad_indoor_dir + item_name +'/digit/'))
                    len_image = len(os.listdir(tacquad_indoor_dir + item_name +'/img_digit/'))
                    time_point = t / (len_touch*1.0)
                    vision_id = int(time_point * len_image)
                    
                    image_path = tacquad_indoor_dir + item_name +'/img_digit/' + str(vision_id) +'.png'

                    self.vision_list.append(image_path)
                    self.touch_list.append(png_path)
                    self.text_list.append(tacquad_text_dir + 'tacquad_indoor_' + str(now_id) +'.pt')
                    self.sensor_list.append(1)

                for t in range(duragel_start, duragel_end+1):

                    png_path = tacquad_indoor_dir + item_name +'/duragel/' + str(t) +'.png'

                    len_touch = len(os.listdir(tacquad_indoor_dir + item_name +'/duragel/'))
                    len_image = len(os.listdir(tacquad_indoor_dir + item_name +'/img_duragel/'))
                    time_point = t / (len_touch*1.0)
                    vision_id = int(time_point * len_image)
                    
                    image_path = tacquad_indoor_dir + item_name +'/img_duragel/' + str(vision_id) +'.png'

                    self.vision_list.append(image_path)
                    self.touch_list.append(png_path)
                    self.text_list.append(tacquad_text_dir + 'tacquad_indoor_' + str(now_id) +'.pt')
                    self.sensor_list.append(4)
                
                now_id += 1

        with open(tacquad_outdoor_file,'r') as file:
            csv_reader = csv.reader(file)
            now_id = 0
            for row in csv_reader:
                item_name = row[0]

                gelsight_start = int(row[1])
                gelsight_end = int(row[2])

                digit_start = int(row[3])
                digit_end = int(row[4])

                duragel_start = int(row[5])
                duragel_end = int(row[6])

                for t in range(gelsight_start, gelsight_end+1):

                    png_path = tacquad_outdoor_dir + item_name +'/gelsight/' + str(t) +'.png'

                    len_touch = len(os.listdir(tacquad_outdoor_dir + item_name +'/gelsight/'))
                    len_image = len(os.listdir(tacquad_outdoor_dir + item_name +'/img_gelsight/'))
                    time_point = t / (len_touch*1.0)
                    vision_id = int(time_point * len_image)
                    
                    image_path = tacquad_outdoor_dir + item_name +'/img_gelsight/' + str(vision_id) +'.png'

                    self.vision_list.append(image_path)
                    self.touch_list.append(png_path)
                    self.text_list.append(tacquad_text_dir + 'tacquad_outdoor_' + str(now_id) +'.pt')
                    self.sensor_list.append(3)
                
                for t in range(digit_start, digit_end+1):

                    png_path = tacquad_outdoor_dir + item_name +'/digit/' + str(t) +'.png'

                    len_touch = len(os.listdir(tacquad_outdoor_dir + item_name +'/digit/'))
                    len_image = len(os.listdir(tacquad_outdoor_dir + item_name +'/img_digit/'))
                    time_point = t / (len_touch*1.0)
                    vision_id = int(time_point * len_image)
                    
                    image_path = tacquad_outdoor_dir + item_name +'/img_digit/' + str(vision_id) +'.png'

                    self.vision_list.append(image_path)
                    self.touch_list.append(png_path)
                    self.text_list.append(tacquad_text_dir + 'tacquad_outdoor_' + str(now_id) +'.pt')
                    self.sensor_list.append(1)

                for t in range(duragel_start, duragel_end+1):

                    png_path = tacquad_outdoor_dir + item_name +'/duragel/' + str(t) +'.png'

                    len_touch = len(os.listdir(tacquad_outdoor_dir + item_name +'/duragel/'))
                    len_image = len(os.listdir(tacquad_outdoor_dir + item_name +'/img_duragel/'))
                    time_point = t / (len_touch*1.0)
                    vision_id = int(time_point * len_image)
                    
                    image_path = tacquad_outdoor_dir + item_name +'/img_duragel/' + str(vision_id) +'.png'

                    self.vision_list.append(image_path)
                    self.touch_list.append(png_path)
                    self.text_list.append(tacquad_text_dir + 'tacquad_outdoor_' + str(now_id) +'.pt')
                    self.sensor_list.append(4)

                now_id += 1



        # if args.train_num_samples is None:
        #     args.train_num_samples = len(os.listdir(os.path.join(self.data_root, "touch")))
        # print(args.train_num_samples)

        # for i in range(args.train_num_samples):
        #     self.touch_list.append(f"image_{i}_tac.jpg")
        #     self.vision_list.append(f"image_{i}_rgb.jpg")
            
        #     self.ids = self.id2title_folder_caps[:args.train_num_samples]
        # else:
        #     self.ids = self.id2title_folder_caps

        self.tokenizer = get_tokenizer(HF_HUB_PREFIX + args.model, cache_dir=args.cache_dir)
        self.vision_transform = get_vision_transform(args)
        self.touch_transform = get_touch_transform(args)

    def __len__(self):
        return len(self.touch_list)

    def __getitem__(self, idx):
        try:
            sent_output, phra_output= self.get_text(idx)
            sent_input_ids, sent_attention_mask = sent_output
            phra_input_ids, phra_attention_mask = sent_output

            matched_modality_touch, matched_modality_vision = self.get_touch_vision(idx)

            # return matched_modality_touch['pixel_values'], matched_modality_vision['pixel_values'], sent_input_ids, sent_attention_mask, phra_input_ids, phra_attention_mask        
            return (
                matched_modality_touch['pixel_values'].float(),
                matched_modality_vision['pixel_values'].float(),
                sent_input_ids.long(),
                sent_attention_mask.long(),
                phra_input_ids.long(),
                phra_attention_mask.long(),
                self.get_sensor(idx)
            )

        except Exception as error_msg:
            logging.info(f"Failed at {idx} with \"{error_msg}\"")
            return self.__getitem__(random.randint(0, self.__len__()-1))


    def get_text(self, id):
        if self.text_list[id] == -1:
            sent_output = (torch.zeros(77).int(), torch.zeros(77).int())
        else:
            sent_output = torch.load(self.text_list[id])
        phra_output = None
        return sent_output, phra_output
    
    def get_sensor(self, id):
        sensor_id = self.sensor_list[id]
        return sensor_id
    
    def get_touch_vision(self, id):
        touch_path = self.touch_list[id]
        touch = load_and_transform_touch(touch_path, self.vision_transform)

        vision_path = self.vision_list[id]
        vision = load_and_transform_vision(vision_path, self.vision_transform)

        return touch,vision
    