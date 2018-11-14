import argparse
import logging
import time

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

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
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

    # initialization
    frame_idx = 0
    total_frame_num = 5
    sliding_frame_num = 2
    body_parts_num = 18
    actors = []
    joints_list = [] # np.zeros(shape=(100, body_parts_num, 3), dtype=np.float32)

    # first load #total_frame_num-1 frames
    for i in range(total_frame_num - 1):
        frame_idx += 1

        ret_val, image = cam.read()
        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        # choose one actor with maximum score
        scores = []
        for human_idx in range(len(humans)):
            score = humans[human_idx].score
            scores.append(score)
        max_human_idx = scores.index(max(scores))
        actor = humans[max_human_idx]
        actors.append(actor)

    while True:

        frame_idx += 1

        ret_val, image = cam.read()
        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        # choose one actor with maximum score
        scores = []
        for human_idx in range(len(humans)):
            score = humans[human_idx].score
            scores.append(score)
        max_human_idx = scores.index(max(scores))
        actor = humans[max_human_idx]
        actors.append(actor)

        # average joints locations over #total_frame_num frames
        # with steps of #sliding_frame_num frames
        if not (frame_idx-total_frame_num) % sliding_frame_num:

            avg_joints = []
            show_human = Human([])

            for p_idx in range(body_parts_num):
            # for p_idx in sorted(body_parts.keys()):
                joint = np.zeros(3, dtype=np.float32)
                idx_sum = 0

                for i in range(frame_idx - total_frame_num, frame_idx):
                    if p_idx not in actors[i].body_parts.keys():
                        continue
                    body_part = actors[i].body_parts[p_idx]
                    joint += np.array([body_part.x, body_part.y, body_part.score])
                    idx_sum += 1

                if idx_sum:
                    avg_joint = joint / idx_sum
                    # show_human with average joints locations
                    show_human.body_parts[p_idx] = BodyPart(
                        '%d-%d' % (0, p_idx), p_idx,
                        avg_joint[0], avg_joint[1], avg_joint[2]
                    )
                else:
                    avg_joint = joint

                avg_joints.append(avg_joint)
                # avg_joints_array = np.array(avg_joints, dtype=np.float32)

            # show image with joints
            logger.debug('postprocess+')
            image = TfPoseEstimator.draw_humans(image, [show_human], imgcopy=False)
            logger.debug('show+')
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
            logger.debug('finished+')

        # save joints locations
        joints_list.append(avg_joints)
        # joints_list_array = np.array(joints_list, dtype=np.float32)


        # logger.debug('postprocess+')
        # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        #
        # logger.debug('show+')
        # cv2.putText(image,
        #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
        #             (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 255, 0), 2)
        # cv2.imshow('tf-pose-estimation result', image)
        # fps_time = time.time()
        # if cv2.waitKey(1) == 27:
        #     break
        #
        # logger.debug('finished+')

    cv2.destroyAllWindows()
