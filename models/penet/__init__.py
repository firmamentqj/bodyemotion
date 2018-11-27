from .input import Dataset
from .pemlpnet import PEMLPNET
from .configuration import configuration as config

def train( dataset, is_reuse ):
    model = PEMLPNET(config, is_reuse, is_train=True, name=config.network_name)
    pose_train = Dataset( dataset )
    model.train(pose_train)

    return pose_train

def train_val( dataset_train, dataset_validation, is_reuse ):
    model = PEMLPNET(config, is_reuse, is_train=True, name=config.network_name)
    pose_train = Dataset( dataset_train )
    pose_validation = Dataset( dataset_validation )
    acc_validation = model.train_val(pose_train, pose_validation)

    return acc_validation

def online_test(pose, is_reuse):
    model = PEMLPNET(config, is_reuse, is_train=False, name=config.network_name)
    prediction, score, CLASS2EMOTION = model.test( pose )


    return prediction, score, CLASS2EMOTION
