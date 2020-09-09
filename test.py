from options import test_options
from dataloader import data_loader
from model import create_model
from util import visualizer
from itertools import islice
import os
import shutil
from shutil import copyfile
from util import util
import glob

if __name__=='__main__':
    # get testing options
    opt = test_options.TestOptions().parse()
    # creat a dataset
    ##create input images directory
    opt.img_file = 'nose_test'
    util.mkdir(opt.img_file)
    if not os.path.exists(opt.img_file):
        os.makedirs(opt.img_file)
    else:
        for f in glob.glob(os.path.join(opt.img_file, '*')):
            os.remove(f)
    copyfile(opt.image1, os.path.join(opt.img_file,os.path.split(opt.image1)[-1]))
    copyfile(opt.image2, os.path.join(opt.img_file,os.path.split(opt.image2)[-1]))
    dataset = data_loader.dataloader(opt,True)
    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    model.eval()
    # create a visualizer
    visualizer = visualizer.Visualizer(opt)

    for i, data in enumerate(islice(dataset, opt.how_many)):
        model.set_input(data)
        model.test()
