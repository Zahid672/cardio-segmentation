import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
import random
from pathlib import Path

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

def prepare_cfg(path):
    f = open(path, 'r')
    info = f.read().splitlines()

    data_dict = {}
    for item in info:
        key, value = item.split(': ')
        data_dict[key] = value
    
    return data_dict

class Prepare_CAMUS():
    def __init__(self, data_path, save_path) -> None:
        super().__init__()
        self.data_path = data_path
        self.save_path = save_path

    def prepare(self):
        if Path(os.path.join(self.save_path, 'train_samples.npy')).is_file() and Path(os.path.join(self.save_path, 'test_ED.npy')).is_file():
            print('\n', '----- The files already exist -----')
        else:
            self.separate_patients(self.data_path)
            self.separate_train_test()

    def create_rep(self, orig_array, mode = 'combined'):
        if mode == 'combined':
            rep_array = []
            for sample in orig_array:
                rep_array.append([sample, 'ED'])
                rep_array.append([sample, 'ES'])
            return rep_array
        
        elif mode == 'separate':
            rep_array_1 = []
            rep_array_2 = []
            for sample in orig_array:
                rep_array_1.append([sample, 'ED'])
                rep_array_2.append([sample, 'ES'])
            return rep_array_1, rep_array_2


    def separate_train_test(self):
        np.random.shuffle(self.high_quality_patients)
        np.random.shuffle(self.medium_quality_patients)
        np.random.shuffle(self.low_quality_patients)

        h_len = len(self.high_quality_patients)
        m_len = len(self.medium_quality_patients)
        l_len = len(self.low_quality_patients)
        h_samps = 20#int(h_len*0.2)
        m_samps = 20#int(m_len*0.2)
        l_samps = 10#int(l_len*0.2)

        htest = self.high_quality_patients[0:h_samps]
        htrain = self.high_quality_patients[h_samps:h_len]

        mtest = self.medium_quality_patients[0:m_samps]
        mtrain = self.medium_quality_patients[m_samps:m_len]

        ltest = self.low_quality_patients[0:l_samps]
        ltrain = self.low_quality_patients[l_samps:l_len]

        htest = np.array(htest)
        mtest = np.array(mtest)
        ltest = np.array(ltest)
        
        total_test = np.array(np.concatenate((htest, mtest, ltest)))
        total_training = np.array(np.concatenate((htrain, mtrain, ltrain)))

        total_test_ED, total_test_ES = self.create_rep(total_test, mode = 'separate')
        total_training = self.create_rep(total_training)

        train_save_path = os.path.join(self.save_path, 'train_samples.npy')
        test_ED_path = os.path.join(self.save_path, 'test_ED.npy')
        test_ES_path = os.path.join(self.save_path, 'test_ES.npy')
        np.save(train_save_path, total_training)
        np.save(test_ED_path, total_test_ED)
        np.save(test_ES_path, total_test_ES)


    def separate_patients(self, folder_path):
        self.high_quality_patients = []
        self.medium_quality_patients = []
        self.low_quality_patients = []

        for patient in os.listdir(folder_path):
            patient_folder = os.path.join(folder_path, patient)
            # print('patient folder: ', patient_folder)
            info_dict = prepare_cfg(os.path.join(patient_folder, 'Info_2CH.cfg'))
            quality = info_dict['ImageQuality']

            if quality == 'Good':
                self.high_quality_patients.append(patient)
            if quality == 'Medium':
                self.medium_quality_patients.append(patient)
            if quality == 'Poor':
                self.low_quality_patients.append(patient)

        print('High quality patients: ', len(self.high_quality_patients))
        print('Medium quality patients: ', len(self.medium_quality_patients))
        print('Low quality patients: ', len(self.low_quality_patients))

def sitk_load(filepath: str):
    """Loads an image using SimpleITK and returns the image and its metadata.

    Args:
        filepath: Path to the image.

    Returns:
        - ([N], H, W), Image array.
        - Collection of metadata.
    """
    # Load image and save info
    image = sitk.ReadImage(str(filepath))
    info = {"origin": image.GetOrigin(), "spacing": image.GetSpacing(), "direction": image.GetDirection()}

    # Extract numpy array from the SimpleITK image object
    im_array = np.squeeze(sitk.GetArrayFromImage(image))

    return im_array, info

def decode_seg_map(image, channels = 4):
        label_colors = np.array([
            (0, 0, 0), # None
            (255, 0, 0), # Endocardium
            (0, 255, 0), # Epicardium
            (0, 0, 255), # Left atrium wall
            ])
    
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        
        for l in range(0, channels):
            r[image==l] = label_colors[l,0]
            g[image==l] = label_colors[l,1]
            b[image==l] = label_colors[l,2]
            
        rgb = np.stack([r, g, b], axis=2)

        return rgb

def save_high_quality_data():
    image_save_path = 'G:\\Dr_zahid\\cardiac_good_samples\\images'
    gt_save_path = 'G:\\Dr_zahid\\cardiac_good_samples\\gt'

    prep_object = Prepare_CAMUS('E:\\CAMUS_public\\database_nifti', None)
    prep_object.separate_patients('E:\\CAMUS_public\\database_nifti')
    hq_patients = prep_object.high_quality_patients

    image_pattern = "{patient_name}_{view}_{instant}.nii.gz"
    gt_mask_pattern = "{patient_name}_{view}_{instant}_gt.nii.gz"

    for i in range(len(hq_patients)):
        patient_name = hq_patients[i]
        print('patient name: ', patient_name)
        patient_dir = os.path.join('E:\\CAMUS_public\\database_nifti', patient_name)

        image, info_image = sitk_load(filepath = os.path.join(patient_dir, image_pattern.format(patient_name = patient_name, view = '2CH', instant = 'ED')))
        gt, info_gt = sitk_load(filepath = os.path.join(patient_dir, gt_mask_pattern.format(patient_name = patient_name, view = '2CH', instant = 'ED')))

        cropped_gt = decode_seg_map(np.where(image == 0, 0, gt))

        print('path: ', os.path.join(image_pattern.format(patient_name = patient_name, view = '2CH', instant = 'ED'))[:-7])
        # print('image/gt shape: ', image.shape, cropped_gt.shape)
        # plt.imshow(image, cmap = 'gray')
        # plt.show()
        # plt.imshow(cropped_gt)
        # plt.show()

        # image_name = os.path.join(image_pattern.format(patient_name = patient_name, view = '2CH', instant = 'ED'))[:-7] + '_image.png'
        # gt_name = os.path.join(image_pattern.format(patient_name = patient_name, view = '2CH', instant = 'ED'))[:-7] + '_GT.png'
        # cv2.imwrite(os.path.join(image_save_path, image_name), image)
        # cv2.imwrite(os.path.join(gt_save_path, gt_name), cropped_gt)

        break


# save_high_quality_data()



def save_array():
    data_array = []
    for sample in os.listdir('G:\\Dr_zahid\\cardiac_good_samples\\images'):
        print(sample)
        name = sample[:-17]
        print('name: ', name)

        data_array.append([name, 'ED'])
        # break

    save_path = 'G:\\Dr_zahid\\cardiac_segmentation\\data_record\\good_samples.npy'
    data_array = np.array(data_array)
    np.save(save_path, data_array)

save_array()

