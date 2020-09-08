import argparse
import os
import torch
import model
from util import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self, parser):
        # base define
        ##change the default name argument to nose
        parser.add_argument('--name', type=str, default='nose', help='name of the experiment.')
        parser.add_argument('--model', type=str, default='pluralistic', help='name of the model type. [pluralistic]')
        ##change to 4, the nose mask type
        parser.add_argument('--mask_type', type=int, default=[4],
                            help='mask type, 0: center mask, 1:random regular mask, '
                            '2: random irregular mask. 3: external irregular mask. 4: nose4 mask. [0],[1,2],[1,2,3]')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are save here')
        parser.add_argument('--which_iter', type=str, default='latest', help='which iterations to load')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0, 1, 2 use -1 for CPU')

        # data pattern define
        parser.add_argument('--img_file', type=str, default='/home/dcor/ronmokady/workshop20/celebA', help='training and testing dataset')
        parser.add_argument('--mask_file', type=str, default='none', help='load test mask')
        ##add new argument, path to annotations file
        parser.add_argument('--ann_file', type=str, default='/home/dcor/ronmokady/workshop20/team5/Nose_Swapping_Finish/ann_file.txt', help='load annotations')
        parser.add_argument('--loadSize', type=int, default=[266, 266], help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=[256, 256], help='then crop to this size')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|]')
        ##change the default flip argument to false
        parser.add_argument('--no_flip', action='store_false', help='if specified, do flip the image for data augmentation')
        parser.add_argument('--no_rotation', action='store_true', help='if specified, do not rotation for data augmentation')
        parser.add_argument('--no_augment', action='store_true', help='if specified, do not augment the image for data augmentation')
        parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
        parser.add_argument('--nThreads', type=int, default=8, help='# threads for loading data')
        parser.add_argument('--no_shuffle', action='store_true',help='if true, takes images serial')

        # display parameter define
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        parser.add_argument('--display_id', type=int, default=1, help='display id of the web')
        parser.add_argument('--display_port', type=int, default=8097, help='visidom port of the web display')
        parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visidom web panel')

        return parser

    def gather_options(self):
        """Add additional model-specific options"""

        if not self.initialized:
            parser = self.initialize(self.parser)

        # get basic options
        opt, _ = parser.parse_known_args()

        # modify the options for different models
        model_option_set = model.get_option_setter(opt.model)
        parser = model_option_set(parser, self.isTrain)
        opt = parser.parse_args()

        return opt

    def parse(self):
        """Parse the options"""

        opt = self.gather_options()
        opt.isTrain = self.isTrain

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids):
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt

        return self.opt

    @staticmethod
    def print_options(opt):
        """print and save options"""

        print('--------------Options--------------')
        for k, v in sorted(vars(opt).items()):
            print('%s: %s' % (str(k), str(v)))
        print('----------------End----------------')

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        if opt.isTrain:
            file_name = os.path.join(expr_dir, 'train_opt.txt')
        else:
            file_name = os.path.join(expr_dir, 'test_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('--------------Options--------------\n')
            for k, v in sorted(vars(opt).items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('----------------End----------------\n')
