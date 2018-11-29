import argparse
import logging
import time
import os
import os.path as op
from signal import SIGINT, SIGTERM

import cv2
import numpy as np
import tensorflow as tf

from tf_pose.estimator import TfPoseEstimator, Human, BodyPart
from tf_pose.networks import get_graph_path, model_wh
import models.penet as model
from models.penet.configuration import configuration as config
from utils import lbtoolbox as lb


EMOTION = ['Neutral', 'Angry', 'Fearful', 'Sad', 'Happy']

WORD2CLASS={'NE': 0,
            'AN': 1,
            'FE': 2,
            'SA': 3,
            'HA': 4}

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
    parser.add_argument('--camera', type=int, default=0, help='choose camera ID')
    parser.add_argument('--resize', type=str, default='368x368',
                        help='if provided, resize images before they are processed. Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
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

    body_parts_num = 18
    frame_idx = 0
    flag_reuse = False

    with open(op.join(config.root_dir, 'results', 'output.txt'), 'w') as fout, lb.Uninterrupt(sigs=[SIGINT, SIGTERM], verbose=True) as u:

        fout.write("\"EMOTION\": CONFIDENCE\n\n")
        while True:

            frame_idx += 1
            fout.write("Frame #{}\n".format(frame_idx))

            ret_val, image = cam.read()
            logger.debug('image process+')
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

            tmp_size = 0

            if not humans:

                # predictEmotion = 'NO human detected!'
                # logger.debug('show+')
                # cv2.putText(image,
                #             "Frame#{}    FPS: {}    Emotion: {}".format(frame_idx, 1.0 / (time.time() - fps_time),
                #                                                         predictEmotion),
                #             (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                #             (255, 255, 0), 2)
                # # cv2.imshow('tf-pose-estimation result', image)
                # fps_time = time.time()
                # if cv2.waitKey(1) == 27:
                #     break
                # logger.debug('finished+')
                continue

            else:
                cv2.putText(image,
                            'Frame#{}  FPS: {}'.format(frame_idx, 1.0 / (time.time() - fps_time)),
                            (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.99, (255, 255, 0), 2)
                for human in humans:
                    joint_list = []
                    for p_idx in range(body_parts_num):
                        joint = np.zeros(2, dtype=np.float32)
                        if p_idx in human.body_parts.keys():
                            body_part = human.body_parts[p_idx]
                            joint += np.array([body_part.x, body_part.y])
                        joint_list.append(joint)
                    joint_array = np.array(joint_list)

                    ## bounding box locations
                    x_array = joint_array[:, 0]
                    y_array = joint_array[:, 1]
                    min_x = np.min(x_array[np.nonzero(x_array)])
                    max_x = np.max(x_array)
                    min_y = np.min(y_array[np.nonzero(y_array)])
                    miny_idx = np.argmin(y_array[np.nonzero(y_array)])
                    max_y = np.max(y_array)

                    ## test module
                    joint_array=np.reshape(joint_array, [1,36])
                    input_test = tf.placeholder('float32', shape=(None, 36))
                    # if frame_idx == 1:
                    #     flag_reuse = False
                    # else:
                    #     flag_reuse = True

                    predictEmotion, confidence, CLASS2EMOTION = model.online_test(joint_array, is_reuse = flag_reuse)
                    flag_reuse = True

                    # if frame_idx==1:
                    #     fout.write( "\"EMOTION\": CONFIDENCE\n\n" )
                    fout.write("{")
                    for kkk in range(len(confidence)):
                        if kkk > 0:
                            print
                            fout.write( " \"%s\":%.06f"%(CLASS2EMOTION[kkk],confidence[kkk]) )
                        else:
                            fout.write( "\"%s\":%.06f"%(CLASS2EMOTION[kkk],confidence[kkk]) )
                    fout.write("}\n")


                    logger.debug('postprocess+')
                    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

                    logger.debug('show+')

                    cv2.putText(image,
                                'Emotion: {}'.format(predictEmotion),
                                (int(x_array[np.nonzero(x_array)][miny_idx]*image.shape[1]), int(min_y*image.shape[0])),  cv2.FONT_HERSHEY_SIMPLEX,
                                0.99, (255, 255, 0), 2)


                    if cv2.waitKey(1) == 27:
                        break
                    logger.debug('finished+')

                    if u.interrupted:
                        fout.close()

                fout.write("\n")
                cv2.imshow('pose-based emotion result', image)
                fps_time = time.time()

        cv2.destroyAllWindows()
    fout.close()