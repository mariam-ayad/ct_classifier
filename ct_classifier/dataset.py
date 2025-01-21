'''
    PyTorch dataset class for COCO-CT-formatted datasets. Note that you could
    use the official PyTorch MS-COCO wrappers:
    https://pytorch.org/vision/master/generated/torchvision.datasets.CocoDetection.html

    We just hack our way through the COCO JSON files here for demonstration
    purposes.

    See also the MS-COCO format on the official Web page:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''

import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Lambda#,#RandomCrop
from kornia.augmentation import Resize,CenterCrop
import pandas as pd
import rasterio
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore",category=UserWarning,module='rasterio')

#warnings.filterwarnings('ignore',message='*.rasterio._env:*')

# class NormalizeSat:
#     def __init__(self,lower_percentile=2, upper_percentile=98 ):
#         self.lower_percentile=lower_percentile
#         self.upper_percentile = upper_percentile
#         # Normalize by clipping the values between the lower and upper percentiles and then scaling to [0, 1]
#         
#         # for band in img:# check this bit!!!!!!!
#         lower = np.percentile(band, self.lower_percentile)
#         upper = np.percentile(band, self.upper_percentile)
#         # Normalize by clipping the values between the lower and upper percentiles and then scaling to [0, 1]
#         normalized_band = np.clip((band - lower) / (upper - lower), 0, 1)
#         


class BleachDataset(Dataset):

    def __init__(self, cfg, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split
        # Transforms. Here's where we could add data augmentation 
        #  For now, we just resize the images to the same dimensions...and convert them to torch.Tensor.
        #  For other transformations see Björn's lecture on August 11 or 
        self.transform = Compose([CenterCrop(tuple(cfg['image_size'])), Lambda(lambda x: x/cfg['normalization_factor'])])
        
        # load annotation file
        annoPath = os.path.join(
            self.data_root,
            #'train_annotations.json' if self.split=='train' else 'cis_val_annotations.json'
            'train.csv' if self.split=='train' else 'val.csv'
        )
        #meta = json.load(open(annoPath, 'r')) #read in csv
        self.meta = pd.read_csv(annoPath)  # Load the CSV file as a DataFrame


        ##############
        sites = self.meta.site.unique()
        self.image_couples = []
        self.couple_labels = []
        for site in sites:
            site_df = self.meta[self.meta.site==site]
            locations = site_df.filename.unique()
            for location_name in locations:
                #location_name = locations[0]
                location_df = site_df[site_df.filename == location_name]
                healthy_df = location_df[location_df.label=='healthy']
                for i, healthy_entry in healthy_df.iterrows(): # go over all healthy image within a specific location within a specific site
                    for j, query_entry in location_df.iterrows(): # match them with any other image from that location in that site (aside from themselves)
                        if healthy_entry.image_id != query_entry.image_id:
                            new_couple = (healthy_entry.image_id, query_entry.image_id) # create new image pair of one healthy image and another query
                            new_label = query_entry.label
                            self.image_couples.append(new_couple)
                            self.couple_labels.append(new_label)

        # enable filename lookup. Creates image IDs and assigns each ID one filename. 
        #  If your original images have multiple detections per image, this code assumes
        #  that you've saved each detection as one image that is cropped to the size of the
        #  detection, e.g., via megadetector.
        # images = dict([[i['id'], i['file_name']] for i in meta['images']])
        # # create custom indices for each category that start at zero. Note: if you have already
        # #  had indices for each category, they might not match the new indices.
        # labels = dict([[c['id'], idx] for idx, c in enumerate(meta['categories'])])
        
        # # since we're doing classification, we're just taking the first annotation per image and drop the rest
        # images_covered = set()      # all those images for which we have already assigned a label
        # for anno in meta['annotations']:
        #     imgID = anno['image_id']
        #     if imgID in images_covered:
        #         continue
            
        #     # append image-label tuple to data
        #     imgFileName = images[imgID]
        #     label = anno['category_id']
        #     labelIndex = labels[label]
        #     self.data.append([imgFileName, labelIndex])
        #     images_covered.add(imgID)       # make sure image is only added once to dataset
    

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.image_couples)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        couple = self.image_couples[idx]
        str_label = np.array(self.couple_labels[idx])
        label = torch.tensor(str_label=='bleached',dtype=torch.float32)
        idxs = couple
        imgs = []
        for idx in idxs:
            filepath = self.meta.query('image_id==@idx').filepath.values[0]
            with rasterio.open(filepath, mode='r') as ds:
                img = ds.read()
                ##*****change the config param for projection issue in rasterio********
            imgs.append(img)
        # transform: see lines 31ff above where we define our transformations
        imgs = torch.tensor(np.concatenate(imgs),dtype=torch.float32)
        img_tensor = self.transform(imgs).squeeze()

        return img_tensor, label