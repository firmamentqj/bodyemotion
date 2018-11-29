import argparse
import logging
import time
import os
import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator, Human, BodyPart
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='368x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--person', type=str, default='',
                        help='person_id is missing: 01=Jie 02=Jiaxi 03=Ionut')
    parser.add_argument('--emotion', type=str, default='',
                        help='NE=Neutral, HA=Happy, SA=Sad, FE=Fearful, AN=Angry, SU=Surprised')
    parser.add_argument('--seqid', type=str, default='')

    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    ############### initialization ################
    body_parts_num = 18
    frame_idx = 0
    image_folder = 'M' + args.person + args.emotion + '_SEQ' + args.seqid
    # path = './dataset_new'
    ##############################################

    if not os.path.exists('./dataset_new/' + image_folder):
        os.mkdir('./dataset_new/' + image_folder)

    while True:
        ret_val, image = cam.read()

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        # show image with joints
        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(image,
                    "Frame#{}    FPS: {} ".format(frame_idx, 1.0 / (time.time() - fps_time)),
                    (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.99,
                    (255, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

        logger.debug('finished+')


    while frame_idx<50:

        frame_idx += 1

        ret_val, image = cam.read()
        # cv2.imshow("capture", image)
        image_idx = '%04d' % frame_idx
        image_file = os.path.join('./dataset_new/' + image_folder, image_folder + '_' + image_idx + '.jpg')
        cv2.imwrite(image_file, image)
        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        joint_list = []
        for p_idx in range(body_parts_num):
            joint = np.zeros(3, dtype=np.float32)

            if p_idx in humans[0].body_parts.keys():
                body_part = humans[0].body_parts[p_idx]
                joint += np.array([body_part.x, body_part.y, body_part.score])
            joint_list.append(joint)

        joint_array = np.array(joint_list)
        joint_file = image_file[:-3] + 'npy'
        np.save(joint_file, joint_array)

        # show image with joints
        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(image,
                    "Frame#{}    FPS: {} ".format(frame_idx, 1.0 / (time.time() - fps_time)),
                    (40, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.99,
                    (255, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

        logger.debug('finished+')

    cv2.destroyAllWindows()
