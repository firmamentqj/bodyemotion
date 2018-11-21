import tensorflow as tf
slim = tf.contrib.slim


def PEMLPNET(input, numb_class, is_training, is_reuse, name=None):
    net = slim.utils.convert_collection_to_dict(('none', 1))
    net['input'] = input
    net['mlp1'] = slim.fully_connected(
        net['input'], 18, reuse=is_reuse, scope='mlp1' )
    net['mlp2'] = slim.fully_connected(
        net['mlp1'], 9, reuse=is_reuse, scope='mlp2' )
    net['logits'] = slim.fully_connected(
        net['mlp2'], numb_class, activation_fn=None, reuse=is_reuse, scope='logits')
    net['prediction'] = tf.nn.softmax( net['logits'] )
    return net

def loss_entropy(predictions,labels, numb_classes):

    val_loss = tf.losses.softmax_cross_entropy(labels, predictions, label_smoothing=0)

    return val_loss
