import copy
import logging
import os
import glob
import torch.utils.data
import torchvision
import pandas as pd
from PIL import Image
from ast import literal_eval
import torchvision.transforms.functional as TF
import cv2
import numpy as np

from . import transforms, utils


ANNOTATIONS_TRAIN = '/data/data-mscoco/annotations/person_keypoints_train2017.json'
ANNOTATIONS_VAL = '/data/data-mscoco/annotations/person_keypoints_val2017.json'
IMAGE_DIR_TRAIN = '/data/data-mscoco/images/train2017/'
IMAGE_DIR_VAL = '/data/data-mscoco/images/val2017/'


def collate_images_anns_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    anns = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return images, anns, metas


def collate_multiscale_images_anns_meta(batch):
    """Collate for multiscale.

    indices:
        images: [scale, batch , ...]
        anns: [batch, scale, ...]
        metas: [batch, scale, ...]
    """
    n_scales = len(batch[0][0])
    images = [torch.utils.data.dataloader.default_collate([b[0][i] for b in batch])
              for i in range(n_scales)]
    anns = [[b[1][i] for b in batch] for i in range(n_scales)]
    metas = [b[2] for b in batch]
    return images, anns, metas


def collate_images_targets_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return images, targets, metas

###############################################################################
###############################################################################
############################# JAAD DATASET ####################################
###############################################################################
###############################################################################
class JAAD(torch.utils.data.Dataset):
    def __init__(self, args, dtype):
        
        # read annotations  
        df = pd.DataFrame()
        for file in glob.glob(os.path.join(args.jaad_dataset,dtype,"*")):
            df = df.append(pd.read_csv(file), ignore_index=True) 
            
        # if input was passed as a single text file, then it means that it has already been processed
        # -------------------------------------------------------------------------------------------
        if("singletxt" in dtype):
            for v in list(df.columns.values):
                df.loc[:,v] = df.loc[:, v].apply(lambda x: literal_eval(x))
                
        # if input was not passed as a single text file, then we have to preprocess it
        # ---------------------------------------------------------------------------
        if("singletxt" not in dtype):
            # assign unique id to each pedestrian
            df["unique_id"] = df.groupby(['imagefolderpath']).ngroup()   
                        
            # keep only those that are not under occlusion
            df = df[df["occlusion"] == 0].reset_index(drop=True)      
    
            # retain sequence for each pedestrian up till he begins to cross 
            # ==============================================================
            if(args.truncate):                   
                ind = []       
                # for each pedestrian
                for i in df["unique_id"].unique():
                    ind_temp = []
                    df_temp = df[df["unique_id"] == i]  
                                
                    # A) pedestrian interacts with driver but does not cross 
                    # retain all of his indices
                    # - cannot only check the last element because ground truth describes the STATE the pedestrian is currently in
                    # - todo: i dont remember why i needed 2 conditions 
                    if((len(np.where(df_temp['crossing_true']>0)[0])==0) and (len(df_temp[df_temp['crossing_true']==1])==0)):                     
                        ind_temp = list(df_temp.index.values)
                        ind = ind + ind_temp
                        continue
                                                    
                    # B) pedestrian interacts with driver and crosses
                    # retain the indices up till pedestrian begins to cross
                    ind_temp = list(df_temp.iloc[0:np.min(np.where(df_temp['crossing_true']>0))+1].index.values)
                    ind = ind + ind_temp  
                    
                # update dataframe                                  
                df = df.iloc[ind].reset_index(drop=True)
                df["unique_id"] = df.groupby(['imagefolderpath']).ngroup() 
                # ==============================================================   
                                        
            # For the pedestrians that cross, convert the labels X timesteps before the crossing, from 0 to 1
            # ==============================================================
            for i in df["unique_id"].unique():
                df_temp = df[df["unique_id"] == i]
                crossing = df_temp["crossing_true"].tolist()
                
                # find indices at which the labels transition from 0 to 1
                indices = match_sublist(crossing,[0,1])
                
                # convert the labels args.final_frame_offset before the crossing, from 0 to 1
                for ind in indices:
                    crossing[max(0,ind-args.final_frame_offset):ind] = [1]*min(ind, args.final_frame_offset)
                    #crossing[ind-args.final_frame_offset:ind] = [1]*args.final_frame_offset
                                                           
                # update dataframe
                df.loc[df["unique_id"] == i, "crossing_true"] = crossing  
                
            # Assign unique ID to all pedestrians that exist together in a scene at any given frame
            df["unique_id"] = df.groupby(['scenefolderpath','filename']).ngroup()                  
            df = df.groupby("unique_id").agg(lambda x: list(x))
                                                                                            
        self.df = df.copy()
        self.args = args
                    
        self.dtype = dtype
        print(dtype, " loaded")
        
        self.transform = tr_transforms
        #if(self.dtype == "train"):
        #    self.transform = tr_transforms
        #if(self.dtype == "val"):
        #    self.transform = tr_transforms

    def __len__(self):
        return len(self.df)
    
    # -------------------------------
    # retrieves one sample
    def __getitem__(self, index):

        # !!!! gotta get by folder,
        df = self.df.iloc[index]
        
        #meta = {}
        # prepare bbox meta data
        # =============================================================   
        # filenames
        #imagepaths = df["imagefolderpath"]
        #scenepaths = df["scenefolderpath"]
        #filenames  = df["filename"]
        #                                        
        ## load scene
        #scenename = os.path.join(df["scenefolderpath"][-1], df["filename"][-1]) 
        ## decision
        #labels = df["crossing_true"]    
        ## image dimensions
        #scene_h = df["im_h"][0]
        #scene_w = df["im_w"][0]
        #scale = scene_h / 540             
        ## bbox dimensions
        #box_x = (np.array(df["x"])/scale).astype(int)
        #box_y = (np.array(df["y"])/scale - int(0.3 * 540)).astype(int)
        #box_w = (np.array(df["w"])/scale).astype(int)
        #box_h = (np.array(df["h"])/scale).astype(int)
        #meta = {
        #    'dataset_index': index,
        #    'path_to_scene': scenename,
        #    'labels': labels,
        #    'box_x': box_x.astype(int),
        #    'box_y': box_y.astype(int),
        #    'box_w': box_w.astype(int),
        #    'box_h': box_h.astype(int)
        #    #'file_name': image_info['file_name'],
        #}
        # ============================================================= 
                
        # load labels
        standing  = df["standing"]             
        looking   = df["looking"]
        walking   = df["walking"]
        crossing  = df["crossing_true"]
        heights   = df["h"]
        
        # image dimensions
        # list containing the exact same numbers
        scene_h = df["im_h"]
        scene_w = df["im_w"]
        scale = scene_h[0] / 540
                
        # box
        box_x = df["x"]
        box_y = df["y"]
        box_w = df["w"]
        box_h = df["h"]
                
        # filenames
        imagepaths = df["imagefolderpath"]
        scenepaths = df["scenefolderpath"]
        filenames  = df["filename"]
        
        meta = {
            'dataset_index': index,
            'path_to_scene': os.path.join(df["scenefolderpath"][-1], df["filename"][-1]),
            'pedestrian': imagepaths,
            'frame': filenames,
            'labels': crossing,
            'box_x': (np.array(box_x)/scale).astype(int),
            'box_y': (np.array(box_y)/scale - int(0.3 * 540)).astype(int),
            'box_w': (np.array(box_w)/scale).astype(int),
            'box_h': (np.array(box_h)/scale).astype(int)
        }
                        
        # load scene
        scenename = os.path.join(df["scenefolderpath"][-1], df["filename"][-1])
        scene = Image.open(scenename)
        scene = self.transform(scene, self.args.crop)
                
        # build activity map
        activity_map, loss_mask = build_activity_map(scene_h, scene_w, self.args.activity_h, self.args.activity_w, box_x, box_y, box_w, box_h, crossing, self.args) 
        activity_map = [[activity_map]] # cheap fix to make the format similar to pifpaf
                
        return scene, activity_map, meta

def build_activity_map(scene_h, scene_w, activity_h, activity_w, box_x, box_y, box_w, box_h, crossing, args):

    # ===============================================================================
    # build the activity map
    # ===============================================================================
    activity_map = np.zeros((2, activity_h, activity_w),dtype=np.float32)
    activity_map_noncrosser = []
    activity_map_crosser = []
    for x, y, w, h, sh, sw, c in zip(box_x, box_y, box_w, box_h, scene_h, scene_w, crossing):
    
        # compute offset for bounding box
        # compute size of scene
        offset = sh * args.crop
        y  -= offset
        sh -= offset
    
        # mu and sigma
        x0, y0, sigma_x, sigma_y = x+float(w)/2, y+float(h)/2, float(w)/4, float(h)/4
        
        # activity map for current person
        y, x = np.arange(sh), np.arange(sw)    
        gy = np.exp(-(y-y0)**2/(2*sigma_y**2))
        gx = np.exp(-(x-x0)**2/(2*sigma_x**2))
        g  = np.outer(gy, gx)     
                        
        if(c==0):
            activity_map_noncrosser.append(np.copy(g))
        if(c==1):
            activity_map_crosser.append(np.copy(g))
    
    if(len(activity_map_noncrosser)!=0):
        activity_map_noncrosser = np.amax(activity_map_noncrosser, axis=0) 
        activity_map[0] = cv2.resize(activity_map_noncrosser, (activity_w,activity_h))
    if(len(activity_map_crosser)!=0):
        activity_map_crosser = np.amax(activity_map_crosser, axis=0) 
        activity_map[1] = cv2.resize(activity_map_crosser, (activity_w,activity_h))
    # ===============================================================================
    loss_mask = np.ones((2, activity_h, activity_w), dtype=np.float32)
        
    return torch.tensor(activity_map).float(), torch.tensor(loss_mask).float()

# -------------------------------
# transforms for the training set
def tr_transforms(scene, crop):

    width = 960
    height = 540

    # transform scene
    # resize to the "standard resolution" before cropping
    scene = TF.resize(scene, size=(height,width))
    scene = TF.to_tensor(scene) 
    scene = TF.normalize(scene, mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    
    # crop
    offset = int(crop * height)
    scene = scene[:,offset:,:]    
    return scene

def te_transforms(scene, crop):

    width = 960#1920
    height = 540#1080

    # transform scene
    # resize to the "standard resolution" before cropping
    scene = TF.resize(scene, size=(height,width))
    scene = TF.to_tensor(scene) 
    scene = TF.normalize(scene, mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    
    # crop
    offset = int(crop * height)
    scene = scene[:,offset:,:]    
    return scene

def match_sublist(x, y):
    indices = []
    l1, l2 = len(x), len(y)
    for i in range(l1):
        if x[i:i+l2] == y:
            indices.append(i+1)
    return indices

###############################################################################
###############################################################################
############################# COCO DATASET ####################################
###############################################################################
###############################################################################

class CocoKeypoints(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Based on `torchvision.dataset.CocoDetection`.

    Caches preprocessing.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, image_transform=None, target_transforms=None,
                 n_images=None, preprocess=None, all_images=False, all_persons=False):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)

        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        if all_images:
            self.ids = self.coco.getImgIds()
        elif all_persons:
            self.ids = self.coco.getImgIds(catIds=self.cat_ids)
        else:
            self.ids = self.coco.getImgIds(catIds=self.cat_ids)
            self.filter_for_keypoint_annotations()
        if n_images:
            self.ids = self.ids[:n_images]
        print('Images: {}'.format(len(self.ids)))

        self.preprocess = preprocess or transforms.Normalize()
        self.image_transform = image_transform or transforms.image_transform
        self.target_transforms = target_transforms

        self.log = logging.getLogger(self.__class__.__name__)

    def filter_for_keypoint_annotations(self):
        print('filter for keypoint annotations ...')
        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_keypoint_annotation(image_id)]
        print('... done.')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)
        anns = copy.deepcopy(anns)

        image_info = self.coco.loadImgs(image_id)[0]
        self.log.debug(image_info)
        with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
            image = Image.open(f).convert('RGB')

        meta_init = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
        }

        if 'flickr_url' in image_info:
            _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
            flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
            meta_init['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)

        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns)
        if isinstance(image, list):
            return self.multi_image_processing(image, anns, meta, meta_init)

        return self.single_image_processing(image, anns, meta, meta_init)

    def multi_image_processing(self, image_list, anns_list, meta_list, meta_init):
        return list(zip(*[
            self.single_image_processing(image, anns, meta, meta_init)
            for image, anns, meta in zip(image_list, anns_list, meta_list)
        ]))

    def single_image_processing(self, image, anns, meta, meta_init):
        meta.update(meta_init)

        # transform image
        original_size = image.size
        image = self.image_transform(image)
        assert image.size(2) == original_size[0]
        assert image.size(1) == original_size[1]

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        # if there are not target transforms, done here
        self.log.debug(meta)
        if self.target_transforms is None:
            return image, anns, meta

        # transform targets
        targets = [t(anns, original_size) for t in self.target_transforms]
        return image, targets, meta

    def __len__(self):
        return len(self.ids)


class ImageList(torch.utils.data.Dataset):
    def __init__(self, image_paths, preprocess=None, image_transform=None):
        self.image_paths = image_paths
        self.image_transform = te_transforms #transforms.image_transform#
        self.preprocess = preprocess

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        if self.preprocess is not None:
            image = self.preprocess(image, [])[0]

        original_image = torchvision.transforms.functional.to_tensor(image)
        original_image = original_image[:,324:,:]
        #print(original_image.size())
        image = self.image_transform(image,0.3) ############################################ self.image_transform(image, 0.3)
        #print(image.size())

        return image_path, original_image, image

    def __len__(self):
        return len(self.image_paths)


class PilImageList(torch.utils.data.Dataset):
    def __init__(self, images, image_transform=None):
        self.images = images
        self.image_transform = image_transform or transforms.image_transform

    def __getitem__(self, index):
        pil_image = self.images[index].copy().convert('RGB')
        original_image = torchvision.transforms.functional.to_tensor(pil_image)
        image = self.image_transform(pil_image)

        return index, original_image, image

    def __len__(self):
        return len(self.images)


def train_cli(parser):
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--train-annotations', default=ANNOTATIONS_TRAIN)
    group.add_argument('--train-image-dir', default=IMAGE_DIR_TRAIN)
    group.add_argument('--val-annotations', default=ANNOTATIONS_VAL)
    group.add_argument('--val-image-dir', default=IMAGE_DIR_VAL)
    group.add_argument('--pre-n-images', default=8000, type=int,
                       help='number of images to sampe for pretraining')
    group.add_argument('--n-images', default=None, type=int,
                       help='number of images to sampe')
    group.add_argument('--duplicate-data', default=None, type=int,
                       help='duplicate data')
    group.add_argument('--pre-duplicate-data', default=None, type=int,
                       help='duplicate pre data in preprocessing')
    group.add_argument('--loader-workers', default=1, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size', default=8, type=int,
                       help='batch size')
                       
    # for jaad           
    group.add_argument('--jaad_dataset', default='/data/haziq-data/jaad/annotations', type=str)
    group.add_argument('--final_frame_offset', default=0, type=int)
    group.add_argument('--activity_h', default=47, type=int) #48
    group.add_argument('--activity_w', default=119, type=int) #120
    group.add_argument('--crop', default=0.3, type=float)
    group.add_argument('--jaad_batch_size', default=4, type=int)
    group.add_argument('--jaad_train', default='singletxt_train', type=str)
    group.add_argument('--jaad_val', default='singletxt_val', type=str)
    group.add_argument('--jaad_pre_train', default='singletxt_pre_train', type=str)
    group.add_argument('--truncate', default=0, type=int)
    
def train_factory(args, preprocess, target_transforms, jaad_datasets):
 
    # !!!!!!!!!!!
    # COCO LOADER
    coco_train_data = CocoKeypoints(
        root=args.train_image_dir,
        annFile=args.train_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.n_images,
    )
    if (0):
        coco_train_data = torch.utils.data.ConcatDataset(
            [coco_train_data for _ in range(2)])
    coco_train_loader = torch.utils.data.DataLoader(
        coco_train_data, batch_size=args.batch_size, shuffle=not args.debug,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    coco_val_data = CocoKeypoints(
        root=args.val_image_dir,
        annFile=args.val_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.n_images,
    )
    if args.duplicate_data:
        coco_val_data = torch.utils.data.ConcatDataset(
            [coco_val_data for _ in range(args.duplicate_data)])
    coco_val_loader = torch.utils.data.DataLoader(
        coco_val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    coco_pre_train_data = CocoKeypoints(
        root=args.train_image_dir,
        annFile=args.train_annotations,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=args.pre_n_images,
    )
    if args.pre_duplicate_data:
        coco_pre_train_data = torch.utils.data.ConcatDataset(
            [coco_pre_train_data for _ in range(args.pre_duplicate_data)])
    coco_pre_train_loader = torch.utils.data.DataLoader(
        coco_pre_train_data, batch_size=args.batch_size, shuffle=True,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)
    
    # !!!!!!!!!!!
    # JAAD LOADER                             
    jaad_train_data = JAAD(args,jaad_datasets[0])    
    jaad_train_loader = torch.utils.data.DataLoader(
        jaad_train_data, batch_size=args.jaad_batch_size, shuffle=True,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    jaad_val_data = JAAD(args,jaad_datasets[1])   
    jaad_val_loader = torch.utils.data.DataLoader(
        jaad_val_data, batch_size=args.jaad_batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)
        
    jaad_pre_train_data = JAAD(args,jaad_datasets[2])    
    jaad_pre_train_loader = torch.utils.data.DataLoader(
        jaad_pre_train_data, batch_size=args.jaad_batch_size, shuffle=True,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        collate_fn=collate_images_targets_meta)

    return coco_train_loader, coco_val_loader, coco_pre_train_loader, jaad_train_loader, jaad_val_loader, jaad_pre_train_loader
