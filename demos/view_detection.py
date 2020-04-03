######### This script is by Minghan, to visualize network predicted 3D bboxes 

import os

import matplotlib.pyplot as plt

import sys
script_path = os.path.dirname(__file__)
sys.path.append(os.path.join(script_path, '../src'))

from monopsr.datasets.kitti import obj_utils, calib_utils
from monopsr.visualization import vis_utils

def main():

    ## root path of data and predictions
    root_data = '/media/sda1/datasets/kitti/object/training'
    root_pred = '/home/minghanz/Repos/monopsr/data/outputs/monopsr_model_000/predictions/kitti_predictions_3d/val/0.1/100000/data'

    calib_dir = os.path.join(root_data, 'calib')
    image_dir = os.path.join(root_data, 'image_2')
    label_dir = root_pred

    ## get the list of all samples from predictions
    files = os.listdir(root_pred)
    frame_names = [f.split('.')[0] for f in files]

    ## loop over every sample
    for sample_name in frame_names:
        ## load image, boxes, calib
        ## draw boxes and show/save
        # image_path = os.path.join(root_data, 'image_2', frame+'.png')
        # calib_path = os.path.join(root_data, 'calib', frame+'.txt')
        # pred_path = os.path.join(root_pred, frame+'.txt')

        frame_calib = calib_utils.get_frame_calib(calib_dir, sample_name)
        cam_p = frame_calib.p2

        f, axes = vis_utils.plots_from_sample_name(image_dir, sample_name, 2, 1)

        # Load labels
        obj_labels = obj_utils.read_labels(label_dir, sample_name)
        for obj in obj_labels:

            # Draw 2D and 3D boxes
            vis_utils.draw_obj_as_box_2d(axes[0], obj)
            vis_utils.draw_obj_as_box_3d(axes[1], obj, cam_p)

        plt.show(block=True)


if __name__ == '__main__':
    main()
