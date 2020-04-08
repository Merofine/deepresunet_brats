# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

import mymodel
from metrics import dice_coef, batch_iou, mean_iou, iou_score ,ppv,sensitivity
import losses
from utils import str2bool, count_params
from sklearn.externals import joblib
from hausdorff import hausdorff_distance

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')
    parser.add_argument('--mode', default=None,
                        help='GetPicture or Calculate')

    args = parser.parse_args()

    return args


def main():
    val_args = parse_args()

    args = joblib.load('models/%s/args.pkl' %val_args.name)

    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # create model
    print("=> creating model %s" %args.arch)
    model = mymodel.__dict__[args.arch](args)

    model = model.cuda()

    # Data loading code
    img_paths = glob(r'D:\Project\CollegeDesign\dataset\Brats2018FoulModel2D\testImage\*')
    mask_paths = glob(r'D:\Project\CollegeDesign\dataset\Brats2018FoulModel2D\testMask\*')

    val_img_paths = img_paths
    val_mask_paths = mask_paths

    #train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
    #   train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %args.name))
    model.eval()

    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    if val_args.mode == "GetPicture":

        """
        获取并保存模型生成的标签图
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            with torch.no_grad():
                for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    input = input.cuda()
                    #target = target.cuda()

                    # compute output
                    if args.deepsupervision:
                        output = model(input)[-1]
                    else:
                        output = model(input)
                    #print("img_paths[i]:%s" % img_paths[i])
                    output = torch.sigmoid(output).data.cpu().numpy()
                    img_paths = val_img_paths[args.batch_size*i:args.batch_size*(i+1)]
                    #print("output_shape:%s"%str(output.shape))


                    for i in range(output.shape[0]):
                        wtName = os.path.basename(img_paths[i])
                        overNum = wtName.find(".npy")
                        wtName = wtName[0:overNum]
                        wtName = wtName + "_WT" + ".png"
                        imsave('output/%s/'%args.name + wtName, (output[i,0,:,:]*255).astype('uint8'))
                        tcName = os.path.basename(img_paths[i])
                        overNum = tcName.find(".npy")
                        tcName = tcName[0:overNum]
                        tcName = tcName + "_TC" + ".png"
                        imsave('output/%s/'%args.name + tcName, (output[i,1,:,:]*255).astype('uint8'))
                        etName = os.path.basename(img_paths[i])
                        overNum = etName.find(".npy")
                        etName = etName[0:overNum]
                        etName = etName + "_ET" + ".png"
                        imsave('output/%s/'%args.name + etName, (output[i,2,:,:]*255).astype('uint8'))


            torch.cuda.empty_cache()
        """
        将验证集中的GT numpy格式转换成图片格式并保存
        """
        print("Saving GT,numpy to picture")
        val_gt_path = 'output/%s/'%args.name + "GT/"
        if not os.path.exists(val_gt_path):
            os.mkdir(val_gt_path)
        for idx in tqdm(range(len(val_mask_paths))):
            mask_path = val_mask_paths[idx]
            name = os.path.basename(mask_path)
            overNum = name.find(".npy")
            name = name[0:overNum]
            wtName = name + "_WT" + ".png"
            tcName = name + "_TC" + ".png"
            etName = name + "_ET" + ".png"

            npmask = np.load(mask_path)

            WT_Label = npmask.copy()
            WT_Label[npmask == 1] = 1.
            WT_Label[npmask == 2] = 1.
            WT_Label[npmask == 4] = 1.
            TC_Label = npmask.copy()
            TC_Label[npmask == 1] = 1.
            TC_Label[npmask == 2] = 0.
            TC_Label[npmask == 4] = 1.
            ET_Label = npmask.copy()
            ET_Label[npmask == 1] = 0.
            ET_Label[npmask == 2] = 0.
            ET_Label[npmask == 4] = 1.

            imsave(val_gt_path + wtName, (WT_Label * 255).astype('uint8'))
            imsave(val_gt_path + tcName, (TC_Label * 255).astype('uint8'))
            imsave(val_gt_path + etName, (ET_Label * 255).astype('uint8'))
        print("Done!")

    if val_args.mode == "Calculate":
        """
        计算各种指标:Dice、Sensitivity、PPV
        """
        wt_dices = []
        tc_dices = []
        et_dices = []
        wt_sensitivities = []
        tc_sensitivities = []
        et_sensitivities = []
        wt_ppvs = []
        tc_ppvs = []
        et_ppvs = []
        wt_Hausdorf = []
        tc_Hausdorf = []
        et_Hausdorf = []

        wtMaskList = []
        tcMaskList = []
        etMaskList = []
        wtPbList = []
        tcPbList = []
        etPbList = []

        maskPath = glob("output/%s/" % args.name + "GT\*.png")
        pbPath = glob("output/%s/" % args.name + "*.png")
        if len(maskPath) == 0:
            print("请先生成图片!")
            return

        for idx in range(len(maskPath)):
            if maskPath[idx].find("_TC.") != -1:
                tcMaskList.append(maskPath[idx])
                tcPbList.append(pbPath[idx])
            elif maskPath[idx].find("_ET.") != -1:
                etMaskList.append(maskPath[idx])
                etPbList.append(pbPath[idx])
            else:
                wtMaskList.append(maskPath[idx])
                wtPbList.append(pbPath[idx])

        for idx in tqdm(range(len(wtMaskList))):
            mask = imread(wtMaskList[idx])
            pb = imread(wtPbList[idx])
            mask = mask.astype('float32') / 255
            pb = pb.astype('float32') / 255

            dice = dice_coef(pb,mask)
            wt_dices.append(dice)
            ppv_n = ppv(pb,mask)
            wt_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(mask, pb)
            wt_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(pb,mask)
            wt_sensitivities.append(sensitivity_n)

        for idx in tqdm(range(len(tcMaskList))):
            mask = imread(tcMaskList[idx])
            pb = imread(tcPbList[idx])
            mask = mask.astype('float32') / 255
            pb = pb.astype('float32') / 255

            dice = dice_coef(pb,mask)
            tc_dices.append(dice)
            ppv_n = ppv(pb,mask)
            tc_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(mask, pb)
            tc_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(pb, mask)
            tc_sensitivities.append(sensitivity_n)

        for idx in tqdm(range(len(etMaskList))):
            mask = imread(etMaskList[idx])
            pb = imread(etPbList[idx])
            mask = mask.astype('float32') / 255
            pb = pb.astype('float32') / 255

            dice = dice_coef(pb,mask)
            et_dices.append(dice)
            ppv_n = ppv(pb,mask)
            et_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(mask, pb)
            et_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(pb, mask)
            et_sensitivities.append(sensitivity_n)

        print('WT Dice: %.4f' % np.mean(wt_dices))
        print('TC Dice: %.4f' % np.mean(tc_dices))
        print('ET Dice: %.4f' % np.mean(et_dices))
        print("=============")
        print('WT PPV: %.4f' % np.mean(wt_ppvs))
        print('TC PPV: %.4f' % np.mean(tc_ppvs))
        print('ET PPV: %.4f' % np.mean(et_ppvs))
        print("=============")
        print('WT sensitivity: %.4f' % np.mean(wt_sensitivities))
        print('TC sensitivity: %.4f' % np.mean(tc_sensitivities))
        print('ET sensitivity: %.4f' % np.mean(et_sensitivities))
        print("=============")
        print('WT Hausdorff: %.4f' % np.mean(wt_Hausdorf))
        print('TC Hausdorff: %.4f' % np.mean(tc_Hausdorf))
        print('ET Hausdorff: %.4f' % np.mean(et_Hausdorf))
        print("=============")


if __name__ == '__main__':
    main( )
