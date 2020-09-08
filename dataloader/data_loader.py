from PIL import Image, ImageFile
import torchvision.transforms as transforms
import torch.utils.data as data
from .image_folder import make_dataset
from util import task
import random
##add pandas package
import os
import pandas as pd


class CreateDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.img_paths, self.img_size = make_dataset(opt.img_file)
        # provides random file for training and testing
        if opt.mask_file != 'none':
            self.mask_paths, self.mask_size = make_dataset(opt.mask_file)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        # load image
        img, img_path, img_path = self.load_img(index)
        ##get image annotations
        annotations = self.load_annotations(os.path.split(img_path)[-1])
        # load mask
        ##add image annotations parameter
        mask = self.load_mask(img, index, annotations, img_path)
        ##change the nose to grayscale or BGR or RGB
        trans = transforms.ToPILImage()
        trans1 = transforms.ToTensor()
        x = annotations['nose_x'] + 22
        y = annotations['nose_y'] + 5
        range_x = annotations['nose_x'] + 54
        range_y = annotations['nose_y'] + 34
        img = trans(img)
        crop = img.crop((x, y, range_x, range_y))
        choice = random.randint(0, 2)
        if choice == 0:
            im = crop
        elif choice == 1:
            im = crop.convert('L')
        else:
            b, g, r = crop.split()
            im = Image.merge("RGB", (r, g, b))
        img.paste(im, (x, y))
        img = trans1(img)
        return {'img': img, 'img_path': img_path, 'mask': mask}

    def __len__(self):
        return self.img_size

    def name(self):
        return "inpainting dataset"

    def load_img(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_path = self.img_paths[index % self.img_size]
        img_pil = Image.open(img_path).convert('RGB')
        img = self.transform(img_pil)
        img_pil.close()
        return img, img_path, img_path

    ##add function to load annotations file
    def load_annotations(self, img_name):
        df = pd.read_csv(self.opt.ann_file, skiprows=1, header=0, delim_whitespace=True)
        return df.loc[img_name]

    ##get also the annotations of image
    def load_mask(self, img, index, annotations, img_path):
        """Load different mask types for training and testing"""
        mask_type_index = random.randint(0, len(self.opt.mask_type) - 1)
        mask_type = self.opt.mask_type[mask_type_index]

        # center mask
        if mask_type == 0:
            return task.center_mask(img)

        # random regular mask
        if mask_type == 1:
            return task.random_regular_mask(img)

        # random irregular mask
        if mask_type == 2:
            return task.random_irregular_mask(img)

        # external mask from "Image Inpainting for Irregular Holes Using Partial Convolutions (ECCV18)"
        if mask_type == 3:
            if self.opt.isTrain:
                mask_index = random.randint(0, self.mask_size - 1)
            else:
                mask_index = index
            mask_pil = Image.open(self.mask_paths[mask_index]).convert('RGB')
            size = mask_pil.size[0]
            if size > mask_pil.size[1]:
                size = mask_pil.size[1]
            mask_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(10),
                                                 transforms.CenterCrop([size, size]),
                                                 transforms.Resize(self.opt.fineSize),
                                                 transforms.ToTensor()
                                                 ])
            mask = (mask_transform(mask_pil) == 0).float()
            mask_pil.close()
            return mask
        ## add new option 4
        if mask_type == 4:
            return task.nose_mask(img, annotations['nose_x'], annotations['nose_y'], img_path)


def dataloader(opt):
    datasets = CreateDataset(opt)
    dataset = data.DataLoader(datasets, batch_size=opt.batchSize, shuffle=not opt.no_shuffle,
                              num_workers=int(opt.nThreads))

    return dataset


def get_transform(opt):
    """Basic process to transform PIL image to torch tensor"""
    transform_list = []
    osize = [opt.loadSize[0], opt.loadSize[1]]
    fsize = [opt.fineSize[0], opt.fineSize[1]]
    if opt.isTrain:
        if opt.resize_or_crop == 'resize_and_crop':
            transform_list.append(transforms.Resize(osize))
            transform_list.append(transforms.RandomCrop(fsize))
        elif opt.resize_or_crop == 'crop':
            transform_list.append(transforms.RandomCrop(fsize))
        if not opt.no_augment:
            transform_list.append(transforms.ColorJitter(0.0, 0.0, 0.0, 0.0))
        if not opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if not opt.no_rotation:
            transform_list.append(transforms.RandomRotation(3))
    else:
        transform_list.append(transforms.Resize(fsize))

    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)
