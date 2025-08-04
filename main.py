import os
import glob
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import util.io

from torchvision.transforms import Compose
from dpt.models import DPTSegmentationModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from data_loaders import Prepare_CAMUS, CAMUS_loader, ACDC_loader
from Trainer import Main_Trainer
from custom_timm.scheduler import create_scheduler

# from util.main_utils import loss_function

from nnUnet import nnUNet
from CANet.canet import CANetOutput
from extended_nnunet import extended_nnUNet

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',  ########### default lr was 5e-4
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)') # default warmup lr was 1e-6
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)') # default min lr was 1e-5, try 1e-9

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
    return parser

parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

def train_mem_only(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
        # print(name)
        if 'hopfield' in name:
            # print(name)
            param.requires_grad = True

def freeze_mem_only(model):
    for name, param in model.named_parameters():
        param.requires_grad = True
        # print(name)
        if 'hopfield' in name:
            # print(name)
            param.requires_grad = False

prep = Prepare_CAMUS('E:\\CAMUS_public\\database_nifti', 'G:\\Dr_zahid\\cardiac_segmentation\\data_record')#\\train_samples.npy', 'G:\\Dr_zahid\\cardiac_segmentation\\data_record\\test_samples.npy')
prep.prepare()

dataset_path = 'E:\\CAMUS_public\\database_nifti'
train_patient_list = 'G:\\Dr_zahid\\cardiac_segmentation\\data_record\\train_samples.npy'
test_patient_list = 'G:\\Dr_zahid\\cardiac_segmentation\\data_record\\test_ED.npy'
good_samples_list = 'G:\\Dr_zahid\\cardiac_segmentation\\data_record\\good_samples.npy'
view = '2CH'

ACDC_training_list = 'G:\\Dr_zahid\\cardiac_segmentation\\data_record\\ACDC\\training_array.npy'
ACDC_validation_list = 'G:\\Dr_zahid\\cardiac_segmentation\\data_record\\ACDC\\testing_array.npy'

# camus_train_dataset = CAMUS_loader(dataset_path, train_patient_list, view)
# camus_validation_dataset = CAMUS_loader(dataset_path, test_patient_list, view)

ACDC_train_dataset = ACDC_loader(ACDC_training_list)
ACDC_validation_dataset = ACDC_loader(ACDC_validation_list)

train_loader = torch.utils.data.DataLoader(dataset = ACDC_train_dataset, batch_size = 2, shuffle = True, pin_memory = True)#, num_workers=3, prefetch_factor=3) # MNIST = 1000, cifar orig = 4
validation_loader = torch.utils.data.DataLoader(dataset = ACDC_validation_dataset, batch_size = 1, shuffle = False, pin_memory = True)#, num_workers=3, prefetch_factor=3) # MNIST = 2000

model_type = 'dpt_hybrid'
default_models = {
    "dpt_large": "weights/dpt_large-ade20k-b12dca68.pt",
    "dpt_hybrid": "weights/dpt_hybrid-ade20k-53898607.pt",
}

# if model_type == "dpt_large":
#     model = DPTSegmentationModel(
#         4, # num classes
#         path = default_models[model_type],
#         backbone="vitl16_384",
#     )
# elif model_type == "dpt_hybrid":
#     model = DPTSegmentationModel(
#         4,
#         path = default_models[model_type],
#         backbone="vitb_rn50_384",
#     )
from unet.unet_model import UNet

# model = UNet(n_channels = 1, n_classes = 4)#.cuda()
model = nnUNet(in_channels=1, out_channels=4)
# model = CANetOutput(backbone='unet_encoder')
# model = extended_nnUNet(in_channels=1, out_channels=4)

EPOCHS = 60
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 5e-4)
# optimizer = torch.optim.SGD(model.parameters(), 0.001, momentum = 0.9)

lr_scheduler, _ = create_scheduler(args, optimizer)

device = torch.device('cuda')

weight_save_path = 'G:\\Dr_zahid\\cardiac_saves\\weight_saves\\tt'
record_save_path = 'G:\\Dr_zahid\\cardiac_saves\\record_saves\\tt.txt'

# selective weight loading in base_model.py

# model.load_state_dict(torch.load('G:\\Dr_zahid\\cardiac_saves\\weight_saves\\hopfield_only_saves\\best_model.pth'), strict = False)
# model.load_state_dict(torch.load('G:\\Dr_zahid\\cardiac_saves\\weight_saves\\extended_nnUnet\\best_model.pth'), strict = False)
# print('-- loaded weights --')

# train_mem_only(model)
# freeze_mem_only(model)
trainer = Main_Trainer(model, loss_function, optimizer, lr_scheduler, train_loader, validation_loader, device)
trainer.train(EPOCHS, weight_save_path, record_save_path)

# val_loss, val_dice, val_class1_dice, val_class2_dice, val_class3_dice, ind_dsc_list = trainer.test()
# ind_dsc_list = ind_dsc_list[ind_dsc_list[:, 1].argsort()]
# np.save('ind_dice_list.npy', ind_dsc_list)

# array = np.load('G:\\Dr_zahid\\cardiac_segmentation\\ind_dice_list.npy', allow_pickle = True)
# # array = array[array[:, 1].argsort()]
# trainer.display_test_results(camus_validation_dataset, weight_path = 'G:\\Dr_zahid\\cardiac_saves\\weight_saves\\extended_nnUnet\\best_model.pth', display_list = array, result_save_path = 'G:\\Dr_zahid\\cardiac_saves\\visual_saves\\extended_nnUnet')


#################################################################################################################################################
# array = np.load('G:\\Dr_zahid\\cardiac_segmentation\\ind_dice_list.npy', allow_pickle = True)
# print('---- array shape: ', array.shape)
# sorted_array = array[array[:, 1].argsort()]
# for samp in sorted_array:
#     ind, dsc = samp
#     print(ind, ' : ', dsc)
#################################################################################################################################################
