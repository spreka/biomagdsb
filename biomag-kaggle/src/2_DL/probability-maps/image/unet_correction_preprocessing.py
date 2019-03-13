import os
import numpy as np
import pandas as pd
import skimage.io as io
from shutil import copytree


def chk_mkdir(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def find_detection_match(bbox, gt_mask_folder_path, p_overlap=0.5, enlarge=10):
    max_overlap = 0
    max_overlap_mask = ''

    for mask_image_name in os.listdir(gt_mask_folder_path):
        mask = io.imread(os.path.join(gt_mask_folder_path, mask_image_name))
        enlarged_bbox = [np.max([0, bbox[0]-enlarge]), np.max([0, bbox[1]-enlarge]), \
                         np.min([mask.shape[1], bbox[2]+enlarge]), np.min([mask.shape[0], bbox[3]+enlarge])]
        bbox_overlap = np.mean(mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]/255)
        if bbox_overlap > max_overlap:
            max_overlap = bbox_overlap
            max_overlap_mask = mask_image_name

    if max_overlap > p_overlap:
        mask = io.imread(os.path.join(gt_mask_folder_path, max_overlap_mask))
        return mask[enlarged_bbox[1]:enlarged_bbox[3], enlarged_bbox[0]:enlarged_bbox[2]]
    else:
        return np.zeros(shape=(enlarged_bbox[3]-enlarged_bbox[1], enlarged_bbox[2]-enlarged_bbox[0]), dtype='uint8')


def match_all_detections(image_name, gt_folder_path, rcnn_results_path, output_path, enlarge=10):
    bboxes = pd.read_csv(
        os.path.join(rcnn_results_path, image_name + '.csv'),
        index_col=None, header=0, sep=';'
    )
    image = io.imread(os.path.join(gt_folder_path, image_name, 'images', image_name+'.png'))
    gt_mask_folder_path = os.path.join(gt_folder_path, image_name, 'masks')
    rcnn_mask_folder_path = os.path.join(rcnn_results_path, image_name, 'masks')

    for bbox_idx, bbox in bboxes.iterrows():
        enlarged_bbox = [np.max([0, bbox[0] - enlarge]), np.max([0, bbox[1] - enlarge]),
                         np.min([image.shape[1], bbox[2] + enlarge]), np.min([image.shape[0], bbox[3] + enlarge])]

        bbox_output_root = os.path.join(output_path, "%s_%d" % (image_name, bbox_idx))
        bbox_output_image = os.path.join(bbox_output_root, 'images')
        bbox_output_gt_mask = os.path.join(bbox_output_root, 'masks')
        bbox_output_rcnn_mask = os.path.join(bbox_output_root, 'RCNN-masks')
        chk_mkdir(bbox_output_root, bbox_output_gt_mask, bbox_output_image, bbox_output_rcnn_mask)
        gt_detection = find_detection_match(bbox, gt_mask_folder_path, enlarge=enlarge)
        rcnn_detection = find_detection_match(bbox, rcnn_mask_folder_path, enlarge=enlarge)

        io.imsave(
            os.path.join(bbox_output_image, "%s_%d.png" % (image_name, bbox_idx)),
            image[enlarged_bbox[1]:enlarged_bbox[3], enlarged_bbox[0]:enlarged_bbox[2]]
        )
        io.imsave(
            os.path.join(bbox_output_gt_mask, "%s_%d.png" % (image_name, bbox_idx)),
            gt_detection
        )
        io.imsave(
            os.path.join(bbox_output_rcnn_mask, "%s_%d.png" % (image_name, bbox_idx)),
            rcnn_detection
        )


if __name__ == '__main__':
    pass
