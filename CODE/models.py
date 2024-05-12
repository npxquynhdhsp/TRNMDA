import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.backend import l2_normalize, maximum
from params import args

def triplet_loss1(y_true, y_pred, miemb_size = args.miemb_size, diemb_size = args.diemb_size):
    def loss_(y_true, y_pred):
        anchor = y_pred[:, :miemb_size]
        positive = y_pred[:, miemb_size:miemb_size + diemb_size]
        negative = y_pred[:, miemb_size + diemb_size:]

        positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
        negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)

        return maximum(positive_dist - negative_dist + args.alpha, 0.)

    return loss_(y_true, y_pred)

def triplet_loss2(y_true, y_pred, miemb_size = args.miemb_size, diemb_size = args.diemb_size):
    def loss_(y_true, y_pred):
        anchor = y_pred[:, :diemb_size]
        positive = y_pred[:, diemb_size:diemb_size + miemb_size]
        negative = y_pred[:, diemb_size + miemb_size:]

        positive_dist = tf.reduce_mean(tf.square(anchor - positive),
                                       axis=1)
        negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)

        return maximum(positive_dist - negative_dist + args.alpha, 0.)

    return loss_(y_true, y_pred)

def get_sample_triplet1(args, ix,train_or_test, loop_i): #train_or_test = [0, 1] (train) / [3, 4] (test)
    if args.type_eval == 'kfold':
        full_pair = pd.read_csv(args.fi_A + 'L' + str(loop_i) + '/y_4loai_fold' + str(ix + 1) + '.csv', header=None)  # A fold begin from 1
    else:
        full_pair = pd.read_csv(args.fi_A + '/y_4loai_dis' + str(ix) + '.csv', header=None)  # A fold begin from 1
    pair_N = np.argwhere(full_pair.values == train_or_test[0])
    pair_P = np.argwhere(full_pair.values == train_or_test[1])

    pair_N = {'mir': pair_N[:, 0], 'dis': pair_N[:, 1]}
    pair_P = {'mir': pair_P[:, 0], 'dis': pair_P[:, 1]}

    triplets, y = [], []
    for mir_i in range(args.mi_num):
        dis_link_mir_i = pair_P['dis'][pair_P['mir'] == mir_i]
        dis_nolink_mir_i = pair_N['dis'][pair_N['mir'] == mir_i]

        for dis_link in dis_link_mir_i:
            triplets.append((mir_i, dis_link, dis_link))
            y.append(1)

        for dis_nolink in dis_nolink_mir_i:
            triplets.append((mir_i, dis_nolink, dis_nolink))
            y.append(0)
    return triplets, y


def get_sample_triplet2(args, ix, train_or_test, loop_i):  # train_or_test = [0, 1] (train) / [3, 4] (test)
    if args.type_eval == 'kfold':
        full_pair = pd.read_csv(args.fi_A + 'L' + str(loop_i) + '/y_4loai_fold' + str(ix + 1) + '.csv', header=None)  # A fold begin from 1
    else:
        full_pair = pd.read_csv(args.fi_A + '/y_4loai_dis' + str(ix) + '.csv', header=None)  # A fold begin from 1
    pair_N = np.argwhere(full_pair.values == train_or_test[0])
    pair_P = np.argwhere(full_pair.values == train_or_test[1])

    pair_N = {'mir': pair_N[:, 0], 'dis': pair_N[:, 1]}
    pair_P = {'mir': pair_P[:, 0], 'dis': pair_P[:, 1]}

    triplets, y = [], []
    for mir_i in range(args.mi_num):
        dis_link_mir_i = pair_P['dis'][pair_P['mir'] == mir_i]
        dis_nolink_mir_i = pair_N['dis'][pair_N[
                                             'mir'] == mir_i]

        for dis_link in dis_link_mir_i:
            triplets.append((dis_link, mir_i, mir_i))
            y.append(1)

        for dis_nolink in dis_nolink_mir_i:
            triplets.append((dis_nolink, mir_i, mir_i))
            y.append(0)
    return triplets, y

