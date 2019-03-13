import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer_cus import Visualizer
from util import html
import sys

opt = TestOptions().parse()
#opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

#print("Opt: {}", opt)
#sys.exit("Done.")

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.HTMLresults_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('%04d: process image... %s.' % (i, img_path))
    visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)
    visualizer.save_predicted_images_raw(opt.output_dir, visuals, img_path, aspect_ratio=opt.aspect_ratio)

webpage.save()
