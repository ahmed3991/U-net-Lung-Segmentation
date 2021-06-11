#!/sr/bin/env python3
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from shutil import copyfile
from tqdm import tqdm
import argparse

from LungDataset import LungDataset
from model.unet import UNet

import gc


def generate_masks(net,dataloader,device,useful_size,current_size):

    """Iterate over data"""

    print("predict masks and croped images")

    predicted_masks=[]
    data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, sample in data_iter:
        imgs, true_masks = sample['image'], sample['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        # mask_type = torch.float32 if net.n_classes == 1 else torch.long

        with torch.set_grad_enabled(False):
            masks_pred = net(imgs)
            pred = torch.sigmoid(masks_pred) > 0.5
            #print(pred.size())
            pred = torch.squeeze(pred)
            #print(pred.size())
        
        masks = pred.detach().cpu().numpy().astype(np.uint8)

        if useful_size != current_size:

            resized_masks = np.zeros((masks.shape[0],useful_size,useful_size),dtype=np.uint8)

            for i,mask in enumerate(masks):

                mask = (mask*255).astype(np.uint8)

                mask_img = Image.fromarray(mask).resize((useful_size,useful_size),Image.LANCZOS)

                resized_masks[i,:,:] = (np.array(mask_img)/255).astype(np.uint8)

            predicted_masks.append(resized_masks)
        else:
            predicted_masks.append(masks)


    return np.concatenate(predicted_masks, axis=0)



def create_predict_data(path,img_list,out,predicted_masks_array):


    croped_out = os.path.join(out,'croped_lung')

    
    for i,img_name in tqdm(enumerate(img_list)):

        img = Image.open(os.path.join(path,'train/'+img_name)).convert('L')

        mask = (predicted_masks_array[i,:,:]*255).astype(np.uint8)

        mask_img = Image.fromarray(mask).resize(img.size,Image.LANCZOS)

        #mask_img.save(os.path.join(masks_out,'mask_'+img_name))

        croped = np.where(np.array(mask_img) == 0, 0, np.array(img)).astype(np.uint8)

        Image.fromarray(croped).save(os.path.join(croped_out,img_name)) 


def get_args():

    parser = argparse.ArgumentParser(description = "Qata_Covid19 Segmentation" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--path',type=str,default='./data/Qata_COV')
    parser.add_argument('--gpu', type=str, default = '0')
    # arguments for training
    parser.add_argument('--img_size_lung', type = int , default = 512)
    parser.add_argument('--img_size_qata', type = int , default = 224)

    parser.add_argument('--load_lung_model', type=str, default='best_checkpoint.pt', help='.pth file path to load model')
    parser.add_argument('--load_qata_model', type=str, default='best_checkpoint.pt', help='.pth file path to load model')

    parser.add_argument('--out', type=str, default='./dataset')
    return parser.parse_args()


def main():

    args = get_args()

    if ~ os.path.exists(args.out):
        print("path created")
        os.mkdir(args.out)
        #os.mkdir(os.path.join(args.out,'Images'))
        #os.mkdir(os.path.join(args.out,'Ground-truths'))
        #os.mkdir(os.path.join(args.out,'predict_Ground-truths'))
        #os.mkdir(os.path.join(args.out,'original_crop_images'))
        os.mkdir(os.path.join(args.out,'croped_lung'))
    
    # set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # default: '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set model
    lung_model = UNet(n_channels=1, n_classes=1).to(device)

    checkpoint = torch.load(args.load_lung_model)
    lung_model.load_state_dict(checkpoint['model_state_dict'])


    """set img size
        - UNet type architecture require input img size be divisible by 2^N,
        - Where N is the number of the Max Pooling layers (in the Vanila UNet N = 5)
    """

    img_size = args.img_size_lung #default: 224


    # set transforms for dataset
    import torchvision.transforms as transforms
    from my_transforms import RandomHorizontalFlip,RandomVerticalFlip,ColorJitter,GrayScale,Resize,ToTensor
    eval_transforms = transforms.Compose([
        GrayScale(),
        Resize(img_size),
        ToTensor()
    ])

    img_path = os.path.join(args.path,'train')
    img_list = os.listdir(img_path)

    dataset = LungDataset(root_dir = args.path,split=img_list,transforms=eval_transforms,img_size=args.img_size_lung)
    dataloader = DataLoader(dataset = dataset , batch_size=16,shuffle=False)
    
    #create_original_data(args.path,args.out)
    
    masks_lung = generate_masks(lung_model,dataloader,device,args.img_size_lung,args.img_size_lung)

    #from numba import cuda
    #cuda.select_device(0)
    #cuda.close()
    #cuda.select_device(0)

    # set model
    qata_model = UNet(n_channels=1, n_classes=1).to(device)

    checkpoint = torch.load(args.load_qata_model)
    qata_model.load_state_dict(checkpoint['model_state_dict'])

    dataset = LungDataset(root_dir = args.path,split=img_list,transforms=eval_transforms,img_size=args.img_size_qata)
    dataloader = DataLoader(dataset = dataset , batch_size=16,shuffle=False)

    masks_qata = generate_masks(qata_model,dataloader,device,args.img_size_lung,args.img_size_qata)

    print(masks_lung.shape)
    print(masks_qata.shape)

    joined_masks = masks_lung+masks_qata

    del masks_lung,masks_qata
    gc.collect()

    masks = np.where(joined_masks==0,0,1)

    create_predict_data(args.path,img_list,args.out,masks)

    #df = create_annotation(args.path)

    #df.to_csv(os.path.join(args.out,'target.csv'),index=False)


if __name__ == '__main__':

    main()
