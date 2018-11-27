import argparse
import logging
import sys
import time
import os

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import matplotlib.pyplot as plt

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    # parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--path', type=str, default='/Users/nigel/PycharmProjects/tf-openpose/dataset')

    args = parser.parse_args()

    body_parts_num = 18

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(368, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    file_id = 0
    for root, dirs, files in os.walk(args.path):
        for dir in dirs:
            files = os.listdir(os.path.join(root, dir))
            for file in files:
                file_id += 1
                # estimate human poses from a single image !
                if os.path.splitext(file)[1] == '.jpg':
                    print('Processing {} images: {}'.format(file_id, file))
                    image = common.read_imgfile(os.path.join(root, dir, file), None, None)
                    if image is None:
                        logger.error('Image can not be read, path=%s' % args.image)
                        sys.exit(-1)
                    t = time.time()
                    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

                    elapsed = time.time() - t

                    logger.info('inference image: %s in %.4f seconds.' % (file, elapsed))

                    if not humans:
                        continue

                    joint_list = []
                    for p_idx in range(body_parts_num):
                        joint = np.zeros(3, dtype=np.float32)

                        if p_idx in humans[0].body_parts.keys():
                            body_part = humans[0].body_parts[p_idx]
                            joint += np.array([body_part.x, body_part.y, body_part.score])
                        joint_list.append(joint)

                    joint_array = np.array(joint_list)
                    filename_save = os.path.splitext(file)[0]
                    # np.save(os.path.join(root, dir, filename_save + "_cmu.npy"), joint_array)


                    # # show human with joints locations
                    # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
                    #
                    # fig = plt.figure()
                    # a = fig.add_subplot(2, 3, 1)
                    # a.set_title('Result')
                    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    #
                    # bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    # bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)
                    #
                    # # show network output
                    # a = fig.add_subplot(2, 3, 2)
                    # plt.imshow(bgimg, alpha=0.5)
                    # tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
                    # plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
                    # plt.colorbar()
                    #
                    # tmp2 = e.pafMat.transpose((2, 0, 1))
                    # tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
                    # tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)
                    #
                    # a = fig.add_subplot(2, 3, 3)
                    # a.set_title('Vectormap-x')
                    # # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
                    # plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
                    # plt.colorbar()
                    #
                    # a = fig.add_subplot(2, 3, 4)
                    # a.set_title('Vectormap-y')
                    # # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
                    # plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
                    # plt.colorbar()
                    #
                    # a = fig.add_subplot(2, 3, 5)
                    # a.set_title('Cropped Human')
                    crop_axis = np.zeros(4, dtype=np.int)
                    x_array = joint_array[:, 0]
                    y_array = joint_array[:, 1]
                    min_x = np.min(x_array[np.nonzero(x_array)])
                    max_x = np.max(x_array)
                    min_y = np.min(y_array[np.nonzero(y_array)])
                    max_y = np.max(y_array)
                    image_h, image_w = image.shape[:2]
                    top_left_x, top_left_y = int(max(0, min_x * image_w - 50)), \
                                             int(max(0, min_y * image_h - 100))
                    bot_right_x, bot_right_y = int(min(max_x * image_w + 50, image_w)), \
                                               int(min(max_y * image_h + 100, image_h))
                    crop_axis = [top_left_x, top_left_y, bot_right_x, bot_right_y]
                    np.save(os.path.join(root, dir, filename_save + "_crop_axis.npy"), crop_axis)
                    # crop_img = image[top_left_y:bot_right_y, top_left_x:bot_right_x]
                    # plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                    # plt.show()