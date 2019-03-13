import os.path
import numpy as np
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


def random_operation(operator, *args, p=0.5):
    if random.uniform(0, 1) < p:
        return [operator(x) for x in args]
    else:
        return args


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot 
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')
    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h)).resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
        B = AB.crop((w2, 0, w, h)).resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
        # random crops
        #cropped_size = (self.opt.loadSize, self.opt.loadSize)
        #crop_params = transforms.RandomCrop(self.opt.loadSize).get_params(A, cropped_size)
        #A = transforms.functional.crop(A, *crop_params)
        #B = transforms.functional.crop(B, *crop_params)
        # random flips
        #A, B = random_operation(transforms.functional.hflip, A, B)
        #A, B = random_operation(transforms.functional.vflip, A, B)
        # random rotations
        #angle = transforms.RandomRotation((0, 360)).get_params((0, 360))
        #A = transforms.functional.rotate(A, angle)
        #B = transforms.functional.rotate(B, angle)
        
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        #A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        #B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
