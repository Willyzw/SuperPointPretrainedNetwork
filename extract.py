import os
import cv2
import argparse
import numpy as np

from scipy.io import savemat
from tqdm import tqdm
from glob import glob
from demo_superpoint import SuperPointFrontend, PointTracker, VideoStreamer, myjet


def parse_args():
    parser = argparse.ArgumentParser(description='Extract SuperPoint features for images in given folder')
    parser.add_argument('--input_folder', help='Folder containing images', required=True)
    parser.add_argument('--output_folder', help='Folder to output results', required=True)
    parser.add_argument('--img_ext', default='.png', help='Image extension')
    args = parser.parse_args()
    os.makedirs(os.path.join(args.output_folder, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, 'mats'), exist_ok=True)
    return args


def make_plot(img, pts, heatmap, tracks):
    num_pts = pts.shape[1]
    num_tracked = tracks.shape[0]

    # Primary output - Show point tracks overlayed on top of input image.
    out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
    tracks[:, 1] /= float(fe.nn_thresh)  # Normalize track scores to [0,1].
    tracker.draw_tracks(out1, tracks)
    cv2.putText(out1, 'Point tracks {}'.format(num_tracked), (20, 30),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), lineType=16)

    # Extra output -- Show current point detections.
    out2 = (np.dstack((img, img, img)) * 255.).astype('uint8')
    for pt in pts.T:
        pt1 = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
    cv2.putText(out2, 'Raw Point Detections {}'.format(num_pts), (20, 30),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), lineType=16)

    # Extra output -- Show the point confidence heatmap.
    min_conf = 0.001
    heatmap[heatmap < min_conf] = min_conf
    heatmap = -np.log(heatmap)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
    out3 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
    out3 = (out3*255).astype('uint8')
    cv2.putText(out3, 'Raw Point Confidences', (20, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), lineType=16)
    return np.hstack((out1, out2, out3))


def save_mat(output_path, name, pts, desc):
    savemat(os.path.join(output_path, 'mats', name), mdict={'pts': pts, 'desc': desc})


def run(args, images, fe, tracker):
    for i, img_file in enumerate(tqdm(images)):
        img = cv2.imread(img_file, 0).astype(np.float32) / 255.

        # Get points and descriptors.
        pts, desc, heatmap = fe.run(img)

        # Add points and descriptors to the tracker.
        tracker.update(pts, desc)

        # Get tracks for points which were match successfully across all frames.
        tracks = tracker.get_tracks(2)

        # Print and show result image
        img_plot = make_plot(img, pts, heatmap, tracks)
        cv2.imwrite(os.path.join(args.output_folder, "plots", "{:04d}.png".format(i)), img_plot)

        # Save result as .mat file
        name = os.path.basename(img_file).replace(args.img_ext, '.mat')
        save_mat(args.output_folder, name, pts, desc)


if __name__ == "__main__":
    args = parse_args()

    # This class helps load input images from different sources.
    images = glob(os.path.join(args.input_folder, "*{}".format(args.img_ext)))
    print('==> Identify {} images'.format(len(images)))

    print('==> Loading pre-trained network.')
    # This class runs the SuperPoint network and processes its outputs.
    fe = SuperPointFrontend(weights_path='superpoint_v1.pth', nms_dist=4,
                            conf_thresh=0.015, nn_thresh=0.7, cuda=False)
    print('==> Successfully loaded pre-trained network.')

    # This class helps merge consecutive point matches into tracks.
    tracker = PointTracker(5, nn_thresh=fe.nn_thresh)

    print('==> Running Demo.')
    run(args, images, fe, tracker)
