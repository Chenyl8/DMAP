import os,sys
import torch.utils.data 
import torchvision.transforms as transforms
import numpy as np 
import json,csv
import torch 
import pickle
from PIL import Image
from torch.autograd import Variable 

class IQON_dataset(torch.utils.data.Dataset):
    def __init__(self, args, split, transform = None):

        self.imgpath = args.imgpath   # item图片存放的路径
        self.item_img_path = {}  

        self.item_att_label = {'0':[0,0,0,0,0,0,0,0,0,0,0]}  #11
        self.item_att_mask = {'0':[0,0,0,0,0,0,0,0,0,0,0,0]}   #12
        
        self.att_num_dic = {'color':13,'price':4,'brand':5181,'category':62,'variety':21,'material':38,'pattern':16,'design':24,'heel':7,'dress_length':4,'sleeve_length':5}

        self.is_train = split == 'train'
        self.split = split
        self.transform = transform

        item_info = os.path.join(args.datadir, 'item_img_num.csv')
        f = open(item_info)
        csv_read = csv.reader(f)
        for line in csv_read: 
            if line[0] =='user':
                continue

            itemid = line[1]
            itemname = line[2]
            img_path =  line[3].strip()
            price = line[4]
            category = line[5].strip() # large number
            variety = line[6].strip()  # small number
            color_0 = line[7].strip()  # color 0
            color_1 = line[8].strip()
            brand = line[9].strip()
            material = line[10].strip()
            pattern = line[11].strip()
            sleeve_length = line[12].strip()
            dress_length = line[13].strip()
            design = line[14].strip()
            heel = line[15].strip()

            self.item_img_path[itemid] = img_path
            self.item_att_label[itemid] = [int(color_0),int(price),int(brand),int(category),int(variety),int(material),int(pattern),int(design),int(heel),int(dress_length),int(sleeve_length)] 
            self.item_att_mask[itemid] = [1,int(price!='0'),int(brand!='0'),int(category!='0'),int(variety!='0'),int(material!='0'),int(pattern!='0'),int(design!='0'),int(heel!='0'),int(dress_length!='0'),int(sleeve_length!='0'),1]
        
        if os.path.exists(os.path.join(args.datadir, 'partial_mask.npy')):
            self.partial_mask = np.load(os.path.join(args.datadir, 'partial_mask.npy'), allow_pickle=True).item()
        else:
            self.partial_mask = self.get_partial_mask(args)
            np.save(os.path.join(args.datadir, 'partial_mask.npy'), self.partial_mask)

        # load compatibility_train/valid/test
        compatibility_dir = os.path.join(args.datadir, '%s_list.csv' % split)
        with open(compatibility_dir, 'r') as f:
            lines = f.readlines()
         
        self.outfit_list = []
        self.target = []
        for line in lines:
            outfit = []
            data = line.strip().split(',')
            for imid in data[1:]:
                if imid == '0':
                    outfit.append('0')
                else:
                    outfit.append(self.item_img_path[imid])
            self.outfit_list.append(outfit)
            self.target.append(int(data[0]))

    def get_partial_mask(self, args):

        compatibility_dir = os.path.join(args.datadir, 'train_list.csv')
        with open(compatibility_dir, 'r') as f:
            lines = f.readlines()

        train_outfit = []
        for line in lines:
            data = line.strip().split(',')
            for imid in data[1:]:
                if imid != '0' and imid not in train_outfit:
                    train_outfit.append(imid)

        partial_mask = {}
        for _ in train_outfit:
            variety = self.item_att_label[_][4]
            if variety not in partial_mask:
                partial_mask[variety] = self.item_att_mask[_]
            else:
                partial_mask[variety] = [self.item_att_mask[_][i] if self.item_att_mask[_][i] > 0 else partial_mask[variety][i] for i in range(len(partial_mask[variety]))]

        for key in partial_mask.keys():
            temp = partial_mask[key]
            temp = [1 if i > 0 else 0 for i in temp]
            partial_mask[key] = temp 
        
        return partial_mask

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, index):
        out = {'img':[],
               'target':[],
               'att_label':[],
               'att_mask':[],
               'partial_mask':[]}
        for im in self.outfit_list[index]: 
            img_path = os.path.join(self.imgpath, im)
            if im =='0':
                img = torch.zeros(size=[3,224,224])
            else:
                try:
                    img = Image.open(img_path).convert('RGB')
                    if self.transform is not None:
                        img = self.transform(img)
                except Exception as e: # missing images
                    img = torch.zeros(size=[3,224,224])
            i_id = im.split('/')[-1].split('_')[0]   #item id
            out['att_label'].append(self.item_att_label[i_id])
            out['att_mask'].append(self.item_att_mask[i_id])
            out['img'].append(img)
            out['partial_mask'].append(self.partial_mask[self.item_att_label[i_id][4]])

        out['target'].append(self.target[index])
        return out 

