import os
import numpy as np
import cv2
import SimpleITK as sitk
import random
from pathlib import Path
import h5py
import collections
import tqdm as tqdm

# import echonet
import echonet_utils

import pandas
import skimage.draw

import torch
from torch.utils.data import Dataset
import torchvision
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

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_size, mask_ratio, train_mode):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        if not isinstance(mask_size, tuple):
            mask_size = (mask_size,) * 2

        self.height, self.width, _ = input_size
        self.length = 1
        self.mask_h_size, self.mask_w_size = mask_size
        self.num_patches = (self.height//self.mask_h_size) * (self.width//self.mask_w_size) * self.length
        self.empty_image = np.ones((self.num_patches, self.mask_h_size, self.mask_w_size))
        self.num_mask = int(mask_ratio * self.num_patches)
        self.train_mode = train_mode

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        if self.train_mode == 'train':
            select_num_mask = int(self.num_mask * (np.random.randint(20, 80) / 100))
        else:
            select_num_mask = int(self.num_mask * 0.75)

        mask = np.hstack([
            np.ones(self.num_patches - select_num_mask),
            np.zeros(select_num_mask),
        ])
        np.random.shuffle(mask)
        mask = np.expand_dims(mask, (1,2))
        mask = np.repeat(mask, self.mask_h_size, axis=(1))
        mask = np.repeat(mask, self.mask_w_size, axis=(2))
        mask = self.empty_image * mask
        return mask # [196]

import transform as transform
import nibabel as nib

class CardiacNet_Dataset(Dataset):
    def __init__(self, data_path, select_set=['ASD', 'Non-ASD'], is_video=True, is_train=True, is_vaild=False, is_test=False):
        # self.args = args
        self.data_path = data_path
        self.select_set = select_set
        self.is_video = is_video
        self.is_train = is_train

        data_list = list()
        data_dict, self.data_all = self.mapping_data(self.data_path)
        for data_type in select_set:
            data_list+=(list(data_dict[data_type].keys()))

        train_list = random.sample(data_list, int(len(data_list) * 0.1))
        test_vaild_list = list(set(data_list).difference(set(train_list)))
        vaild_list = random.sample(test_vaild_list, int(len(test_vaild_list) * 0.5))
        test_list = list(set(test_vaild_list).difference(set(vaild_list)))

        if self.is_train:
            self.data_select = train_list
        elif self.is_vaild:
            self.data_select = vaild_list
        elif self.is_test:
            self.data_select = test_list

        if is_train:
            self.transform = transforms.Compose([
                                transform.ToTensorVideo(),
                                #transform.NormalizeVideo(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
                                transform.ResizedVideo((144, 144)),
                                transform.RandomCropVideo((112, 112)),
                                transform.RandomHorizontalFlipVideo(),
                                ])
        else:
            self.transform = transforms.Compose([
                                transform.ToTensorVideo(),
                                #transform.NormalizeVideo(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
                                transform.ResizedVideo((144, 144)),
                                transform.CenterCropVideo((112, 112)),
                                ])

        image_size = (112, 112, 16)
        mask_size = 112 #8
        mask_ratio = 0.7
        max_sample_rate = None
        min_sample_rate = None

        self.masked_position_generator = RandomMaskingGenerator(input_size = image_size, mask_size = mask_size, mask_ratio = mask_ratio, train_mode = 'train')

    def __getitem__(self, index):
        image_size = (112, 112, 16) #256
        max_sample_rate = None
        min_sample_rate = None

        if self.is_train:
            index = index // 4
        def get_case_attr(index):
            case_dir = self.data_select[index]
            case_attr = self.data_all[case_dir]
            return case_dir, case_attr

        frame_list = list()
        if self.is_video:
            case_dir, case_attr = get_case_attr(index)

            nii_file = nib.load(case_dir)
            image_data = nii_file.get_fdata()
            print('image data shape: ', image_data.shape)
            w, h, video_length = image_data.shape
            if video_length >= image_size[-1]:
                if self.is_train:
                    if max_sample_rate is not None:
                        max_sample_rate = max_sample_rate
                        if max_sample_rate > video_length // image_size[-1]:
                            max_sample_rate = video_length // image_size[-1]
                    else:
                        max_sample_rate = video_length // image_size[-1]

                    if min_sample_rate is not None:
                        if min_sample_rate < max_sample_rate:
                            min_sample_rate = min_sample_rate
                    else:
                        min_sample_rate = max_sample_rate // 2

                    if max_sample_rate >= 8:
                        sample_rate = random.randint(min_sample_rate, 8)
                    elif max_sample_rate > 4 and max_sample_rate <= 8:
                        sample_rate = random.randint(min_sample_rate, max_sample_rate)
                    elif max_sample_rate > 2 and max_sample_rate <= 4:
                        sample_rate = random.randint(min_sample_rate, max_sample_rate)
                    elif max_sample_rate <= 2 and max_sample_rate > 1:
                        sample_rate = random.randint(1, max_sample_rate)
                    else:
                        sample_rate = 1
                    
                    start_idx = random.randint(0, (video_length-sample_rate * image_size[-1]))

                elif self.is_train is False:
                    sample_rate = 1
                    start_idx = 0
                
                frame_list = image_data[:,:,start_idx : start_idx + sample_rate * image_size[-1] : sample_rate]
            
            # elif video_length < self.args.image_size[-1]:
            #     view_frame_list = image_data
            #     for _ in range(self.args.image_size[-1]-video_length):
            #         view_frame_list.append(np.zeros_like(view_frame_list[-1].shape))
            video = np.transpose(np.array(frame_list), (2, 0, 1))
            bboxs = self.mask_find_bboxs(np.where(video[0] != 0))
            video = video[:, bboxs[0]:bboxs[1], bboxs[2]+15:bboxs[3]]
            video = self.transform(np.expand_dims(video, axis=-1).astype(np.uint8))

            # torchvision.utils.save_image(video, os.path.join("/home/jyangcu/Pulmonary_Arterial_Hypertension/results", f"example_{video.shape[0]}.jpg"), nrow=video.shape[0])
            # print('saved')

            class_type = case_attr['class']

        return video, class_type, self.masked_position_generator()

    def __len__(self):
        if self.is_train:
            return len(self.data_select) * 4
        else:
            return len(self.data_select)

    def mapping_data(self, dataset_path):
        data_all = {}
        data_dict = {'Non-ASD':{}, 'ASD':{}, 'Non-PAH':{},'PAH':{}}
        # Mapping All the Data According to the Hospital Center
        for dir in os.listdir(dataset_path):
            dir_path = os.path.join(dataset_path, dir)
            if os.path.isdir(dir_path):
                data_dict[dir] = {}
                # Mapping All the Data for a Hospital Center According to the Device
                for sub_dir in os.listdir(dir_path):
                    sub_dir_path = os.path.join(dir_path, sub_dir)
                    for case_dir in os.listdir(sub_dir_path):
                        if 'label' in case_dir:
                            pass
                        else:
                            if 'Non-ASD' in sub_dir_path:
                                type_class = 0
                                data_dict['Non-ASD'][sub_dir_path+'/'+case_dir] = {'class':type_class}
                            if 'ASD' in sub_dir_path and 'Non-ASD' not in sub_dir_path:
                                type_class = 1
                                data_dict['ASD'][sub_dir_path+'/'+case_dir] = {'class':type_class}
                            if 'Non-PAH' in sub_dir_path:
                                type_class = 2
                                data_dict['Non-PAH'][sub_dir_path+'/'+case_dir] = {'class':type_class}
                            if 'PAH' in sub_dir_path and 'Non-PAH' not in sub_dir_path:
                                type_class = 3
                                data_dict['PAH'][sub_dir_path+'/'+case_dir] = {'class':type_class}
                            data_all[sub_dir_path+'/'+case_dir] = {'class':type_class}
        return data_dict, data_all

    def mask_find_bboxs(self, mask):
        bbox = np.min(mask[0]), np.max(mask[0]), np.min(mask[1]), np.max(mask[1])
        return bbox

import torch
import matplotlib.pyplot as plt
import cv2


# echo_loader = CardiacNet_Dataset('G:\\Dr_zahid\\CardiacNet')
# print("-- Done loading --")
# train_loader = torch.utils.data.DataLoader(dataset = echo_loader, batch_size = 2, shuffle = True, pin_memory = True)

# for sample in train_loader:
#     # image, target = sample
#     video, class_type, mask = sample
#     # print('len image/target: ', len(video), len(mask))
#     # print('shapes: ', video.shape, mask.shape)
#     # print('class type: ', class_type)

#     image_frame = video[0, :, 1, :, :].numpy().transpose(1, 2, 0)
#     mask_frame = mask[0, 0, :, :].numpy()#.transpose(1, 2, 0)
#     print('image/mask frame shape: ', image_frame.shape, mask_frame.shape)
#     print('mask uniques: ', np.unique(mask), np.unique(mask_frame))
    # cv2.imwrite('test_image_frame.png', image_frame)
    
    # plt.imshow(image_frame)
    # plt.show()
    # plt.imshow(mask_frame)
    # plt.show()

    # break

class Prepare_CardiacNet():
    def __init__(self, data_path, save_path, save_name):
        self.data_path = data_path
        self.save_path = save_path
        self.save_name = save_name
        self.slice_data = []

    def save_data(self):
        self.slice_data = np.array(self.slice_data)
        np.save(os.path.join(self.save_path, self.save_name), self.slice_data)

    def get_slices(self, image_volume, label_volume):
        c, h, w = image_volume
        for s in range(c):
            image_slice = image_volume[s, :, :]
            label_slice = label_volume[s, :, :]
            
            uniques = np.unique(label_volume)
            if len(np.unique(label_slice)) > 3:
                self.slice_data.append([image_slice, label_slice])

    def prepare_dataset(self):
        for cat in os.listdir(self.data_path):
            cat_path = os.path.join(self.data_path, cat)
            for sub_cat in os.listdir(cat_path):
                sub_cat_path = os.path.join(cat_path, sub_cat)
                for file in os.listdir(sub_cat_path):

                    if 'image' in file:
                        mask_name = file.replace('image', 'label')

                        image_path = os.path.join(sub_cat_path, file)
                        mask_path = os.path.join(sub_cat_path, mask_name)

                        image = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))))
                        mask = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))))

                        self.get_slices(image, mask)

        


class CardiacNet_loader(Dataset):
    def __init__(self, data_path):
        super().__init__()

        self.data_array = np.load(data_path, allow_pickle = True)

        self.data_transform = {
            "image": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                # transforms.RandomRotation(degrees = (0, 180)),
                # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]),
            "gt": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), interpolation = InterpolationMode.NEAREST),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])
        }

    def __len__(self):
        return len(self.data_array)
    
    def __getitem__(self, index):
        image_slice, label_slice = self.data_array[index]
        # print('image range: ', np.min(image_slice), np.max(image_slice))

        image_slice = self.data_transform['image'](image_slice*255)
        label_slice = self.data_transform['gt'](label_slice)

        return image_slice, label_slice



class CAMUS_loader(Dataset):
    def __init__(self, data_path, patient_list_path, view):
        super().__init__()
        self.data_path = data_path
        self.patient_list_path = patient_list_path
        self.patient_list_path = np.load(self.patient_list_path)

        assert view in ['2CH', '4CH']
        self.view = view
        self.instants = ['ED', 'ES']

        self.data_transform = {
            "image": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                # transforms.RandomRotation(degrees = (0, 180)),
                # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                # transforms.Normalize((0.1307,), (0.3081,))
            ]),
            "gt": transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224), interpolation = InterpolationMode.NEAREST),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])
        }

    def __len__(self) -> int:
        return len(self.patient_list_path)
    
    def __getitem__(self, index):
        rnum = random.randint(0, 1)

        image_pattern = "{patient_name}_{view}_{instant}.nii.gz"
        gt_mask_pattern = "{patient_name}_{view}_{instant}_gt.nii.gz"

        patient_name, instant = self.patient_list_path[index]
        patient_dir = os.path.join(self.data_path, patient_name)

        image, info_image = self.sitk_load(os.path.join(patient_dir, image_pattern.format(patient_name = patient_name, view = self.view, instant = instant)))
        gt, info_gt = self.sitk_load(os.path.join(patient_dir, gt_mask_pattern.format(patient_name = patient_name, view = self.view, instant = instant)))

        cropped_gt = np.where(image == 0, 0, gt)

        # print('info image: ', info_image)
        # print('info gt: ', info_gt)
        # image = cv2.resize(image, (224, 224))
        # gt = cv2.resize(gt, (224, 224), interpolation = cv2.INTER_NEAREST)
        print('original shapes: ', image.shape, gt.shape)
        print('image range: ', np.min(image), np.max(image))
        image = self.data_transform['image'](image)
        gt = self.data_transform['gt'](cropped_gt)

        return image, gt
    
    def sitk_load(self, filepath: str):
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
        # np.save(train_save_path, total_training)
        # np.save(test_ED_path, total_test_ED)
        # np.save(test_ES_path, total_test_ES)


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



class Process_ACDC():
    def __init__(self, folder_path, save_path):
        self.folder_path = folder_path
        self.save_path = save_path
    
    def save_data_slices(self):
        data_array = []

        for file in os.listdir(self.folder_path):
            with h5py.File(os.path.join(self.folder_path, file)) as f:
                image = f['image']
                label = f['label']

                slices = image.shape[0]
                for slice in range(slices):
                    image_slice = image[slice, :, :]
                    label_slice = label[slice, :, :]

                    label_uniques = np.unique(label_slice)
                    if len(label_uniques) == 3:
                        image_slice = cv2.resize(image_slice, (224, 224))
                        label_slice = cv2.resize(label_slice, (224, 224), interpolation = cv2.INTER_NEAREST)
                        print('image/labels shapes: ', image_slice.shape, label_slice.shape)
                        data_array.append([image_slice, label_slice])

        data_array = np.array(data_array)
        print('Data array shape ', data_array.shape)
        np.save(self.save_path, data_array)


# folder_path = 'E:\\ACDC_datasets\\ACDC_preprocessed\\ACDC_testing_volumes'
# save_path = 'G:\\Dr_zahid\\cardiac_segmentation\\data_record\\ACDC\\testing_array.npy'

# data_processor = Process_ACDC(folder_path, save_path)
# data_processor.save_data_slices()



# Open the file
# with h5py.File(file_path, 'r') as f:
#     # List all groups
#     print("Keys: %s" % f.keys())

#     # Get the data
#     image = f['image']
#     label = f['label']
#     print(image.shape)
#     print(label.shape)
#     i = 1
#     image_slice = image[i, :, :]
#     label_slice = label[i, :, :]

#     print('label uniques: ', np.unique(label_slice))

#     plt.imshow(image_slice)
#     plt.show()
#     plt.imshow(label_slice)
#     plt.show()

image_path = 'G:\\Dr_zahid\\CardiacNet\\CardiacNet-PAH\\PAH\\002_image.nii.gz'
mask_path = 'G:\\Dr_zahid\\CardiacNet\\CardiacNet-PAH\\PAH\\002_label.nii.gz'

image = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))))
mask = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))))


# slices_with_more_than_3_labels = []

# for slice_index in range(mask.shape[0]):
#     slice_data = mask[slice_index]
#     unique_labels = np.unique(slice_data)
    
#     if len(unique_labels) > 3:
#         slices_with_more_than_3_labels.append(slice_index)

# print("Slices with more than 3 unique labels:", slices_with_more_than_3_labels)


i = 14

print('Image/mask shape: ', image.shape, mask.shape)
print('Mask uniques: ', np.unique(mask), np.unique(mask[i, :, :]))

plt.imshow(image[i, :, :])
plt.show()
plt.imshow(mask[i, :, :])
plt.show()