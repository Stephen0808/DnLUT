import os
import random
import sys

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "../")  # run under the project directory
from common.utils import modcrop


class Provider(object):
    def __init__(self, batch_size, num_workers, scale, path, patch_size):
        self.data = DIV2K(scale, path, patch_size)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.is_cuda = True
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return int(sys.maxsize)

    def build(self):
        self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                         shuffle=False, drop_last=False, pin_memory=False))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch[0], batch[1]
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch[0], batch[1]


class ProviderDN(object):
    def __init__(self, batch_size, num_workers, scale, path, patch_size, sigma):
        self.data = DFWB(scale, path, patch_size, sigma)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.is_cuda = True
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return int(sys.maxsize)

    def build(self):
        self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                         shuffle=False, drop_last=False, pin_memory=False))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch[0], batch[1]
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch[0], batch[1]


class ProviderDN_C(object):
    def __init__(self, batch_size, num_workers, scale, path, patch_size, sigma):
        if sigma == 0:
            self.data = SIDD(scale, path, patch_size, sigma)
        else:
            self.data = DFWB_C(scale, path, patch_size, sigma)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.is_cuda = True
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return int(sys.maxsize)

    def build(self):
        self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                         shuffle=False, drop_last=False, pin_memory=False))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch[0], batch[1]
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
            return batch[0], batch[1]

class DFWB(Dataset):
    def __init__(self, scale, path, patch_size, sigma, rigid_aug=True):
        super(DFWB, self).__init__()
        self.scale = 1
        self.sz = patch_size
        self.rigid_aug = rigid_aug
        self.path = path
        self.sigma = sigma
        self.hr_ims = os.listdir(path)

    def __getitem__(self, _dump):
        key = random.choice(range(0,71580))
        lb = np.array(Image.open(os.path.join(self.path,self.hr_ims[key])))
        im = lb.copy()

        shape = im.shape
        i = random.randint(0, shape[0] - self.sz)
        j = random.randint(0, shape[1] - self.sz)
        c = random.choice([0, 1, 2])

        lb = lb[i * self.scale:i * self.scale + self.sz * self.scale,
             j * self.scale:j * self.scale + self.sz * self.scale, c]
        im = im[i:i + self.sz, j:j + self.sz, c]

        if self.rigid_aug:
            if random.uniform(0, 1) < 0.5:
                lb = np.fliplr(lb)
                im = np.fliplr(im)

            if random.uniform(0, 1) < 0.5:
                lb = np.flipud(lb)
                im = np.flipud(im)

            k = random.choice([0, 1, 2, 3])
            lb = np.rot90(lb, k)
            im = np.rot90(im, k)

        lb = lb.astype(np.float32) / 255.0 
        

        im = im.astype(np.float32) / 255.0 
        im = im + np.random.normal(0, self.sigma/255.0, lb.shape)
        lb = np.expand_dims(lb, axis=0)
        im = np.expand_dims(im, axis=0)
        if False:
            import cv2
            img_gt = lb
            img_n = im
            img_gt = (img_gt * 255).astype(np.uint8)
            img_n = (img_n * 255).astype(np.uint8)
            print(img_gt.shape, img_n.shape)
            img_n = img_n.transpose(1,2,0)
            img_gt = img_gt.transpose(1,2,0)
            img = np.hstack((img_n, img_gt))
            cv2.imwrite('output_image_gray.png', img)
        # im = np.clip(im, 0, 1)
        

        return im, lb

    def __len__(self):
        return int(sys.maxsize)


class DFWB_C(Dataset):
    def __init__(self, scale, path, patch_size, sigma, rigid_aug=True):
        super(DFWB_C, self).__init__()
        self.scale = 1
        self.sz = patch_size
        self.rigid_aug = rigid_aug
        self.path = path
        self.sigma = sigma
        self.hr_ims = os.listdir(path)

    def __getitem__(self, _dump):
        key = random.choice(range(0,71580))
        lb = np.array(Image.open(os.path.join(self.path,self.hr_ims[key])))
        im = lb.copy()

        shape = im.shape
        i = random.randint(0, shape[0] - self.sz)
        j = random.randint(0, shape[1] - self.sz)

        lb = lb[i * self.scale:i * self.scale + self.sz * self.scale,
             j * self.scale:j * self.scale + self.sz * self.scale, :]
        im = im[i:i + self.sz, j:j + self.sz, :]

        if self.rigid_aug:
            if random.uniform(0, 1) < 0.5:
                lb = np.fliplr(lb)
                im = np.fliplr(im)

            if random.uniform(0, 1) < 0.5:
                lb = np.flipud(lb)
                im = np.flipud(im)

            k = random.choice([0, 1, 2, 3])
            lb = np.rot90(lb, k)
            im = np.rot90(im, k)

        lb = lb.astype(np.float32) / 255.0 
        

        im = im.astype(np.float32) / 255.0 
        # im = im + np.random.normal(0, self.sigma/255.0, lb.shape)
        im = im + np.random.randn(48, 48, 3) * self.sigma / 255
        im = np.clip(im, 0, 1)

        
        if False:
            import cv2
            img_gt = lb
            img_n = im
            img_gt = (img_gt * 255).astype(np.uint8)
            img_n = (img_n * 255).astype(np.uint8)
            print(img_gt.shape, img_n.shape)
            r_gt, g_gt, b_gt = cv2.split(img_gt)
            r_n, g_n, b_n = cv2.split(img_n)
            img1 = np.hstack((r_gt, g_gt, b_gt))
            img2 = np.hstack((r_n, g_n, b_n))
            img = np.vstack((img1, img2))
            cv2.imwrite('output_image_color.png', img)
        # im = np.clip(im, 0, 1)
        im = im.transpose(2,0,1)
        lb = lb.transpose(2,0,1)
        # lb = np.expand_dims(lb, axis=0)
        # im = np.expand_dims(im, axis=0)

        return im, lb

    def __len__(self):
        return int(sys.maxsize)


class SIDD(Dataset):
    def __init__(self, scale, path, patch_size, sigma, rigid_aug=True):
        super(SIDD, self).__init__()
        self.scale = 1
        self.sz = patch_size
        self.rigid_aug = rigid_aug
        self.path = path
        self.sigma = sigma
        self.gt_ims = os.listdir(os.path.join(path, 'target_crops'))

    def __getitem__(self, _dump):
        key = random.choice(range(0,30608))
        gt = np.array(Image.open(os.path.join(os.path.join(self.path, 'target_crops'),self.gt_ims[key])))
        ns = np.array(Image.open(os.path.join(os.path.join(self.path, 'input_crops'),self.gt_ims[key])))

        shape = gt.shape
        i = random.randint(0, shape[0] - self.sz)
        j = random.randint(0, shape[1] - self.sz)

        gt = gt[i * self.scale:i * self.scale + self.sz * self.scale,
             j * self.scale:j * self.scale + self.sz * self.scale, :]
        ns = ns[i:i + self.sz, j:j + self.sz, :]

        if self.rigid_aug:
            if random.uniform(0, 1) < 0.5:
                gt = np.fliplr(gt)
                ns = np.fliplr(ns)

            if random.uniform(0, 1) < 0.5:
                gt = np.flipud(gt)
                ns = np.flipud(ns)

            k = random.choice([0, 1, 2, 3])
            gt = np.rot90(gt, k)
            ns = np.rot90(ns, k)

        gt = gt.astype(np.float32) / 255.0 
        

        ns = ns.astype(np.float32) / 255.0 

        
        # im = np.clip(im, 0, 1)
        gt = gt.transpose(2,0,1)
        ns = ns.transpose(2,0,1)
        # lb = np.expand_dims(lb, axis=0)
        # im = np.expand_dims(im, axis=0)

        return ns, gt

    def __len__(self):
        return int(sys.maxsize)


class DIV2K(Dataset):
    def __init__(self, scale, path, patch_size, rigid_aug=True):
        super(DIV2K, self).__init__()
        self.scale = scale
        self.sz = patch_size
        self.rigid_aug = rigid_aug
        self.path = path
        self.file_list = [str(i).zfill(4)
                          for i in range(1, 901)]  # use both train and valid


    def __getitem__(self, _dump):
        key = random.choice(self.file_list)
        lb = self.hr_ims[key]
        im = self.lr_ims[key]

        shape = im.shape
        i = random.randint(0, shape[0] - self.sz)
        j = random.randint(0, shape[1] - self.sz)
        c = random.choice([0, 1, 2])

        lb = lb[i * self.scale:i * self.scale + self.sz * self.scale,
             j * self.scale:j * self.scale + self.sz * self.scale, c]
        im = im[i:i + self.sz, j:j + self.sz, c]

        if self.rigid_aug:
            if random.uniform(0, 1) < 0.5:
                lb = np.fliplr(lb)
                im = np.fliplr(im)

            if random.uniform(0, 1) < 0.5:
                lb = np.flipud(lb)
                im = np.flipud(im)

            k = random.choice([0, 1, 2, 3])
            lb = np.rot90(lb, k)
            im = np.rot90(im, k)

        lb = np.expand_dims(lb.astype(np.float32) / 255.0, axis=0)
        im = np.expand_dims(im.astype(np.float32) / 255.0, axis=0)

        return im, lb

    def __len__(self):
        return int(sys.maxsize)


class SRBenchmark(Dataset):
    def __init__(self, path, scale=4):
        super(SRBenchmark, self).__init__()
        self.ims = dict()
        self.files = dict()
        _ims_all = (5 + 14 + 100 + 100 + 109) * 2

        for dataset in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']:
            folder = os.path.join(path, dataset, 'HR')
            files = os.listdir(folder)
            files.sort()
            self.files[dataset] = files

            for i in range(len(files)):
                im_hr = np.array(Image.open(
                    os.path.join(path, dataset, 'HR', files[i])))
                im_hr = modcrop(im_hr, scale)
                if len(im_hr.shape) == 2:
                    im_hr = np.expand_dims(im_hr, axis=2)

                    im_hr = np.concatenate([im_hr, im_hr, im_hr], axis=2)

                key = dataset + '_' + files[i][:-4]
                self.ims[key] = im_hr

                im_lr = np.array(Image.open(
                    os.path.join(path, dataset, 'LR_bicubic/X%d' % scale, files[i])))  # [:-4] + 'x%d.png'%scale)))
                if len(im_lr.shape) == 2:
                    im_lr = np.expand_dims(im_lr, axis=2)

                    im_lr = np.concatenate([im_lr, im_lr, im_lr], axis=2)

                key = dataset + '_' + files[i][:-4] + 'x%d' % scale
                self.ims[key] = im_lr

                assert (im_lr.shape[0] * scale == im_hr.shape[0])

                assert (im_lr.shape[1] * scale == im_hr.shape[1])
                assert (im_lr.shape[2] == im_hr.shape[2] == 3)

        assert (len(self.ims.keys()) == _ims_all)


class DNBenchmark(Dataset):
    def __init__(self, path, sigma):
        super(DNBenchmark, self).__init__()
        self.ims = dict()
        self.files = dict()
        self.sigma_test = sigma
        _ims_all = (68 + 24 + 12 + 100 + 18) * 2

        for dataset in ['CBSD68', 'Kodak', 'Set12', 'Urban100', 'McMaster']:
            folder = os.path.join(path, dataset)
            files = os.listdir(folder)
            files.sort()
            self.files[dataset] = files

            for i in range(len(files)):
                im_hr = np.array(Image.open(
                    os.path.join(path, dataset, files[i])))
                im_hr = modcrop(im_hr, 4)
                if len(im_hr.shape) == 2:
                    im_hr = np.expand_dims(im_hr, axis=2)

                    im_hr = np.concatenate([im_hr, im_hr, im_hr], axis=2)

                key = dataset + '_' + files[i][:-4]
                self.ims[key] = im_hr
                
                np.random.seed(seed=0)
                im_lr = im_hr / 255 + np.random.normal(0, self.sigma_test/255.0, im_hr.shape)
                im_lr = np.clip(im_lr, 0, 1)
                im_lr = np.uint8(im_lr*255)

                key = dataset + '_' + files[i][:-4] + '_noise'
                self.ims[key] = im_lr

                assert (im_lr.shape[0] == im_hr.shape[0])

                assert (im_lr.shape[1] == im_hr.shape[1])
                assert (im_lr.shape[2] == im_hr.shape[2] == 3)

        assert (len(self.ims.keys()) == _ims_all)


class SIDD_VAL(Dataset):
    def __init__(self, path, sigma):
        super(SIDD_VAL, self).__init__()
        self.ims = dict()
        self.files = dict()

        for dataset in ['SIDD']:
            folder_gt = os.path.join(path, dataset, 'target_crops')
            files = os.listdir(folder_gt)
            files.sort()
            self.files[dataset] = files

            for i in range(len(files)):
                im_gt = np.array(Image.open(
                    os.path.join(path, dataset, 'target_crops', files[i])))
                im_gt = modcrop(im_gt, 4)

                im_ns = np.array(Image.open(
                    os.path.join(path, dataset, 'input_crops', files[i])))
                im_ns = modcrop(im_ns, 4)

                key_gt = dataset + '_' + files[i][:-4]
                self.ims[key_gt] = im_gt
                

                key_ns = dataset + '_' + files[i][:-4] + '_noise'
                self.ims[key_ns] = im_ns

                assert (im_gt.shape[0] == im_ns.shape[0])

                assert (im_gt.shape[1] == im_ns.shape[1])
                assert (im_gt.shape[2] == im_ns.shape[2] == 3)


if __name__ == '__main__':
    valid = DN_SIDD('/home/mnt/ysd/SIDD/val', 0)
    print(valid.ims['SIDD_0000-0000_noise'].shape)
    # dataset = DFWB_C(1, '/mnt/data_16TB/ysd21/Denoise/Datasets/train/DFWB', 48, 15)
    # print(dataset.__getitem__(0))

    train = ProviderDN_C(1, 8, 1, '/home/mnt/ysd/SIDD_train', 48, 0)
    print(train.next()[0].shape)