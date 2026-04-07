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
import random
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


# class PairPerImageZScore:
#     def __init__(self, eps=1e-6):
#         self.eps = eps
#     def __call__(self, x):
#         # x: [6,H,W] = [img1_RGB, img2_RGB]
#         x1, x2 = x[:3], x[3:]
#         def norm(z):
#             m = z.mean(dim=(1,2), keepdim=True)
#             s = z.std(dim=(1,2), keepdim=True)
#             return (z - m) / (s + self.eps)
#         return torch.cat([norm(x1), norm(x2)], dim=0)
        
class NormalizeByFactor:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x / self.factor
class BleachDataset(Dataset):

    def __init__(self, cfg, split='train',eval=False):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split
        # Transforms. Here's where we could add data augmentation 
        #  For now, we just resize the images to the same dimensions...and convert them to torch.Tensor.
        #  For other transformations see Björn's lecture on August 11 or 

        # self.transform = Compose([
        #     CenterCrop(tuple(cfg['image_size'])),
        #     NormalizeByFactor(cfg['normalization_factor']),
        #     PairPerImageZScore()
        # ])

        
        self.transform = Compose([
            NormalizeByFactor(cfg['normalization_factor'])
        ])      
        # load annotation file
        annoPath = os.path.join(
            self.data_root,
            #'train_annotations.json' if self.split=='train' else 'cis_val_annotations.json'
            'train.csv' if self.split=='train' else 'val.csv'
        )
        #meta = json.load(open(annoPath, 'r')) #read in csv
        self.meta = pd.read_csv(annoPath)  # Load the CSV file as a DataFrame

        self.image_size = tuple(cfg['image_size'])
       
        self.normalization_factor = float(cfg['normalization_factor'])
        self.augment = (split == 'train') and (not eval)

        self.eval = eval
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
    
    def _augment_geom(self, x, m):
        """
        x: [16,H,W] image channels (two 8-band images concatenated)
        m: [ 2,H,W] mask channels (one per image)
        returns augmented (x, m) with same shapes
        """
        # Horizontal flip
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            m = torch.flip(m, dims=[2])
    
        # Vertical flip
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
            m = torch.flip(m, dims=[1])
    
        # Random 90-degree rotations
        k = random.randint(0, 3)  # 0,1,2,3 -> 0,90,180,270
        if k:
            x = torch.rot90(x, k, dims=[1, 2])
            m = torch.rot90(m, k, dims=[1, 2])
    
        return x, m


    def _augment_photo(self, x):
        """
        Photometric augmentation for multispectral data.
        x: [16,H,W] (two 8-band images concatenated)
        returns x with same shape
        """
        # Split ref/query so we can apply the SAME transform to both (keeps pair consistent)
        x1 = x[:8]
        x2 = x[8:16]
    
        # --- brightness (additive) ---
        if random.random() < 0.8:
            b = random.uniform(-0.08, 0.08)  # in normalized scale (since you divide by 10000)
            x1 = x1 + b
            x2 = x2 + b
    
        # --- contrast (multiplicative) ---
        if random.random() < 0.8:
            c = random.uniform(0.8, 1.2)
            x1 = x1 * c
            x2 = x2 * c
    
        # --- per-band gain jitter (small) ---
        # encourages invariance to band calibration differences across sites
        if random.random() < 0.7:
            gains = torch.empty(8, 1, 1, dtype=x.dtype).uniform_(0.9, 1.1)
            x1 = x1 * gains
            x2 = x2 * gains
    
        # --- clamp to a reasonable range ---
        # (after /10000, most values should be in [0,1-ish], but keep slack)
        x1 = torch.clamp(x1, -0.2, 1.5)
        x2 = torch.clamp(x2, -0.2, 1.5)
    
        return torch.cat([x1, x2], dim=0)


    def _augment_noise(self, x):
        """
        Add small gaussian noise (images only).
        x: [16,H,W]
        """
        if random.random() < 0.5:
            sigma = random.uniform(0.0, 0.02)  # tuned for normalized scale
            x = x + sigma * torch.randn_like(x)
            x = torch.clamp(x, -0.2, 1.5)
        return x


    def _augment_band_dropout(self, x):
        """
        Randomly zero out 1-2 bands sometimes (both ref and query together).
        This prevents reliance on a single band that might shift across sites.
        x: [16,H,W]
        """
        if random.random() < 0.3:
            # drop k bands
            k = 1 if random.random() < 0.7 else 2
            bands = random.sample(range(8), k)
    
            for b in bands:
                x[b, :, :] = 0.0       # ref band b
                x[8 + b, :, :] = 0.0   # qry band b
    
        return x
    
    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.image_couples)

    
    def __getitem__(self, idx):
        couple = self.image_couples[idx]
        str_label = np.array(self.couple_labels[idx])
        label = torch.tensor(str_label=='bleached', dtype=torch.float32)
    
        idxs = couple
        imgs = []
        valid_masks = []
    
        for img_id in idxs:
            filepath = self.meta.query('image_id==@img_id').filepath.values[0]
            with rasterio.Env(GTIFF_SRS_SOURCE="EPSG"):
                with rasterio.open(filepath, mode='r') as ds:
                    img = ds.read().astype(np.float32)  # [B,H,W]
    
            # mask: 1 where all bands are finite (not NaN), else 0
            valid = np.isfinite(img).all(axis=0).astype(np.float32)  # [H,W]
    
            # fill NaNs with 0 for model stability
            img = np.nan_to_num(img, nan=0.0)
    
            imgs.append(img)
            valid_masks.append(valid)
    
        # imgs: list of np arrays [8,H,W] each
        # valid_masks: list of np arrays [H,W] each


        x = np.concatenate(imgs, axis=0).astype(np.float32)      # [16,H,W]
        m = np.stack(valid_masks, axis=0).astype(np.float32)     # [2,H,W]
        
        x = torch.from_numpy(x)                                  # [16,H,W]
        m = torch.from_numpy(m)                                  # [2,H,W]
        
        # --- Center crop in CHW (no Kornia) ---
        H, W = x.shape[1], x.shape[2]
        crop_h, crop_w = self.image_size
        
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2
        
        x = x[:, top:top+crop_h, left:left+crop_w]               # [16,h,w]
        m = m[:, top:top+crop_h, left:left+crop_w]               # [2,h,w]


        if self.augment:
            x, m = self._augment_geom(x, m)

        # normalize images only
        x = x / self.normalization_factor
        
        if self.augment:
            x = self._augment_photo(x)
            x = self._augment_noise(x)
            x = self._augment_band_dropout(x)
        
        # final concat -> [18,h,w]
        img_tensor = torch.cat([x, m], dim=0)
    
        if self.eval:
            return img_tensor, label, couple
        else:
            return img_tensor, label