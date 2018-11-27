import argparse
import logging
import time
import os
import os.path as op

import cv2
import numpy as np
import tensorflow as tf

from tf_pose.estimator import TfPoseEstimator, Human, BodyPart
from tf_pose.networks import get_graph_path, model_wh
import models.penet as model
from models.penet.configuration import configuration as config


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

    # # initialization
    body_parts_num = 18
    frame_idx = 0


    while True:
        with open(op.join(config.root_dir, 'results', 'output.txt'), 'w' ) as fout:

            frame_idx += 1

            ret_val, image = cam.read()
            logger.debug('image process+')
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

            joint_list = []
            for p_idx in range(body_parts_num):
                joint = np.zeros(2, dtype=np.float32)

                if p_idx in humans[0].body_parts.keys():
                    body_part = humans[0].body_parts[p_idx]
                    joint += np.array([body_part.x, body_part.y])
                joint_list.append(joint)

            joint_array = np.array(joint_list)
            # np.save(os.path.join('./results', filename_save + ".npy"), joint_array)
            ## add test module
            joint_array=np.reshape( joint_array, [1,36] )
            input_test = tf.placeholder( 'float32', shape=(None, 36) )
            if frame_idx==1:
                flag_reuse=False
            else:
                flag_reuse=True
            ''' 
            peClassifier = PEMLPNET( input_test, 5, is_training=True, is_reuse=flag_reuse, name='PENET' )
            predictionScore = peClassifier['prediction']
            var_train  = tf.trainable_variables()
            var_penet = [var for var in var_train if 'PENET' in var.name ]
            penet_saver = tf.train.Saver( var_penet )
            with tf.Session() as sess_penet:
                penet_saver.restore(sess_penet, './models/PEMLPNET/checkpoint-100000')
                feed_dict = {input_test: joint_array}
                score = sess_penet.run( predictionScore, feed_dict = feed_dict )
            label = np.argmax(score)
            predict_emotion = EMOTION[label]
            '''
            predictEmotion, confidence, CLASS2EMOTION = model.online_test(joint_array, is_reuse = flag_reuse)
            if frame_idx==1:
                fout.write( "EMOTION\tCONFIDENCE\n\n" )
            fout.write("{")
            for kkk in range(len(confidence)):
                if kkk>0:
                    fout.write( "\t%s:%.06f"%(CLASS2EMOTION[kkk],confidence[kkk]) )
                else:
                    fout.write( "%s:%.06f"%(CLASS2EMOTION[kkk],confidence[kkk]) )
            fout.write("}\n")
            fout.flush()




            logger.debug('postprocess+')
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            logger.debug('show+')
            cv2.putText(image,
                        "Frame#{}    FPS: {}    Emotion: {}".format(frame_idx, 1.0 / (time.time() - fps_time), predictEmotion),
                        (40, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.99,
                        (255, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
            logger.debug('finished+')

        fout.close()
    cv2.destroyAllWindows()