import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import cv2

from util.main_utils import class_dice

class Main_Trainer():
    def __init__(self, model, loss_function, optimizer, lr_scheduler, train_loader, validation_loader, device):
        self.model = model.to(device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.device = device

    def train(self, epochs, weight_save_path, record_save_path):
        dice_latch = 0
        for epoch in range(epochs):
            train_loss = []
            train_dice = []
            train_dice_1 = []
            train_dice_2 = []
            train_dice_3 = []

            # if epoch > 15:
            #     self.optimizer = torch.optim.Adam(self.model.parameters(), 0.0001)

            self.model.train()
            for sample in tqdm(self.train_loader):
                image, gt = sample
                image, gt = image.to(self.device), gt.to(self.device)
                
                # print('---- input shape: ', image.shape)

                gt = torch.squeeze(gt)

                output = self.model(image)

                if image.shape[0] == 1:
                    gt = gt.unsqueeze(dim = 0)
                # print('before loss shapes: ', output.shape, gt.shape)
                loss = self.loss_function(output, gt.long())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                output = torch.argmax(output, dim = 1)
                #print('gt / pred uniques: ', torch.unique(gt), ' ', torch.unique(output))
                dsc, class_dsc, ious, class_iou = class_dice(output, gt, 4)

                train_loss.append(loss.item())
                train_dice.append(dsc.detach().cpu().numpy())
                train_dice_1.append(class_dsc[0])
                train_dice_2.append(class_dsc[1])
                train_dice_3.append(class_dsc[2])

            print('Epoch: ', epoch)
            print('Total Train dice: ', np.mean(train_dice))
            print('Endocardium Train dice: ', np.mean(train_dice_1))
            print('Epicardium Train dice: ', np.mean(train_dice_2))
            print('Left atrium wall Train dice: ', np.mean(train_dice_3))
            print('\n')

            self.lr_scheduler.step(epoch)

            val_loss, val_dice, val_class1_dice, val_class2_dice, val_class3_dice, ind_dsc_list = self.test()

            print('Total Validation dice: ', val_dice)
            print('Endocardium Validation dice: ', val_class1_dice)
            print('Epicardium Validation dice: ', val_class2_dice)
            print('Left atrium wall Validation dice: ', val_class3_dice)

            with open(record_save_path, 'a') as f:
                f.write(f'Epoch: {epoch}')
                f.write('\n')
                f.write(f'Train Loss: {np.mean(train_loss)} Train Dice: {np.mean(train_dice)}')
                f.write('\n')
                f.write(f'Train Endocardium dice: {np.mean(train_dice_1)}')
                f.write('\n')
                f.write(f'Train Epicardium dice: {np.mean(train_dice_2)}')
                f.write('\n')
                f.write(f'Train Left atrium wall dice: {np.mean(train_dice_3)}')
                f.write('\n')
                
                f.write(f'Validation Loss: {val_loss} Validation Dice: {val_dice}')
                f.write('\n')
                f.write(f'Validation Endocardium dice: {val_class1_dice}')
                f.write('\n')
                f.write(f'Validation Epicardium dice: {val_class2_dice}')
                f.write('\n')
                f.write(f'Validation Left atrium wall dice: {val_class3_dice}')
                f.write('\n')
                f.write('\n')
            
            if val_dice > dice_latch:
                dice_latch = val_dice
                # torch.save(self.model.state_dict(), os.path.join(weight_save_path, 'best_model.pth'))
                print('----- Model Saved -----')

    
    def test(self):
        test_loss = []
        test_dice = []
        test_dice_1 = []
        test_dice_2 = []
        test_dice_3 = []

        ind_dice_record = []

        self.model.eval()
        for ind, sample in enumerate(self.validation_loader):
            image, gt = sample
            image, gt = image.to(self.device), gt.to(self.device)
            gt = torch.squeeze(gt, dim = 1)
            # print('---- input shape: ', image.shape)
            with torch.no_grad():
                output = self.model(image.float())

            loss = self.loss_function(output, gt.long())

            output = torch.argmax(output, dim = 1)
            dsc, class_dsc, ious, class_iou = class_dice(output, gt, 4)

            # print(ind, ' : ', dsc.numpy())
            ind_dice_record.append([ind, dsc.numpy()])
            test_loss.append(loss.item())
            test_dice.append(dsc)
            test_dice_1.append(class_dsc[0])
            test_dice_2.append(class_dsc[1])
            test_dice_3.append(class_dsc[2])
        
        # print('test shape: ', )
        return np.mean(test_loss), np.mean(test_dice), np.mean(test_dice_1), np.mean(test_dice_2), np.mean(test_dice_3), np.array(ind_dice_record)

    def decode_seg_map(self, image, channels = 4):
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

    def create_markings(self, image):
        for i in range(224):
            for j in range(224):
                if i%16 == 0 or j%16 == 0:
                    image[i, j] = 255
        
        return image

    def overlay_mask(self, image, mask):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # print('image and mask shape: ', image.shape, mask.shape)
        image = np.array(image,  dtype=np.uint8)
        mask = np.array(mask,  dtype=np.uint8)
        alpha = 0.5
        cv2.addWeighted(mask, alpha, image, 1 - alpha, 0, image)
        return image

    def low_pass_filter(self, image):
        sz = 10
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow-sz:crow+sz, ccol-sz:ccol+sz] = 1
        mask = cv2.bitwise_not(mask) 

        return mask

    def frequency_analysis(self, image):
        dft = cv2.dft(np.float32(image), flags = cv2.DFT_COMPLEX_OUTPUT)
        print('dft shape: ', dft.shape)
        dft_shift = np.fft.fftshift(dft)
        print('shifted dft: ', dft_shift.shape)

        fshift = dft_shift * self.low_pass_filter(image)
        plt.imshow(self.low_pass_filter(image)[:, :, 0])
        plt.show()
        plt.imshow(self.low_pass_filter(image)[:, :, 1])
        plt.show()

        # invert back to image
        f_inv_shift = np.fft.ifftshift(fshift)
        image_back = cv2.idft(f_inv_shift)
        print('image back shape: ', image_back.shape)
        image_back = cv2.magnitude(image_back[:, :, 0], image_back[:, :, 1])

        return image_back

    
    def display_test_results(self, camus_validation_dataset, weight_path, display_list, result_save_path):
        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()

        for ind in range(len(display_list)):
            display_index, dsc = display_list[ind]
            print('index/dice: ', display_index, dsc)
            # display_index = 30
            sample, gt = camus_validation_dataset.__getitem__(int(display_index)) # 6, 7, 13
            # print('sample shape: ', sample.shape)
            # print('gt shape: ', gt.shape)

            image = torch.unsqueeze(sample, dim = 0).to(self.device)
            with torch.no_grad():
                print('====== image shape: ', image.shape)
                output = self.model(image)

            output = torch.squeeze(output)
            output = torch.argmax(output, dim = 0).detach().cpu().numpy()
            # print('output for decode: ', output.shape, gt.shape)

            image = image.squeeze().detach().cpu().numpy()
            gt = gt.squeeze().detach().cpu().numpy()

            output = np.where(image == 0, 0, output)

            # cropped_gt = np.where(image == 0, 0, gt)
            rgb_gt = self.decode_seg_map(gt)
            rgb_output = self.decode_seg_map(output)

            # marked_image = self.create_markings(image)
            overlayed_gt = self.overlay_mask(image, rgb_gt)
            overlayed_pred = self.overlay_mask(image, rgb_output)
            # rgb_output = np.transpose(rgb_output, (2, 0, 1))
            # print('before save shape: ', image.shape, gt.shape, rgb_output.shape)
            image_name = 'Image_' + str(display_index) + '.png'
            pred_name = 'Pred_' + str(display_index) + '.png'
            gt_name = 'GT_' + str(display_index) + '.png'
            # cv2.imwrite(os.path.join(result_save_path, image_name), image)
            # cv2.imwrite(os.path.join(result_save_path, pred_name), np.uint8(overlayed_pred))
            # cv2.imwrite(os.path.join(result_save_path, gt_name), np.uint8(overlayed_gt))
            cv2.imwrite(os.path.join(result_save_path, image_name), np.uint8(image))
            cv2.imwrite(os.path.join(result_save_path, pred_name), np.uint8(overlayed_pred))
            cv2.imwrite(os.path.join(result_save_path, gt_name), np.uint8(overlayed_gt))

            # low_freq = self.frequency_analysis(image)

            # print('Display index: ', display_index)
            # plt.imshow(image)
            # plt.show()
            # plt.imshow(overlayed_gt)
            # plt.show()
            # plt.imshow(overlayed_pred)
            # plt.show()
            # break
        
