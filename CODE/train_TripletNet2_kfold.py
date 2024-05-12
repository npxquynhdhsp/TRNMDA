# %%
from params import args
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as keras_layers
import tensorflow.keras.models as keras_models
import pickle
from sklearn.metrics import auc, roc_auc_score, accuracy_score, precision_recall_curve, precision_score, \
confusion_matrix
from models import triplet_loss2, get_sample_triplet2

# %%
def get_pair(args, fold, loop_i): #get pair indexes from y_4loai_fold
    full_pair = pd.read_csv(args.fi_A + 'L' + str(loop_i) + '/y_4loai_fold' + str(fold+1) +'.csv', header=None) #A fold begin from 1
    pair_trN = np.argwhere(full_pair.values == 0)
    pair_trP = np.argwhere(full_pair.values == 1)
    # pair_teN = np.argwhere(full_pair.values == 3)
    # pair_teP = np.argwhere(full_pair.values == 4)
    print('pair_trainP.shape', pair_trP.shape)
    print('pair_trainN.shape', pair_trN.shape)
    # print('pair_testP.shape', pair_teP.shape)
    # print('pair_testN.shape', pair_teN.shape)
    pair_trN = {'mir': pair_trN[:, 0], 'dis': pair_trN[:, 1]}
    pair_trP = {'mir': pair_trP[:, 0], 'dis': pair_trP[:, 1]}
    # pair_teN = {'mir': pair_teN[:, 0], 'dis': pair_teN[:, 1]}
    # pair_teP = {'mir': pair_teP[:, 0], 'dis': pair_teP[:, 1]}
    # return pair_trP, pair_trN, pair_teP, pair_teN
    return pair_trP, pair_trN

# %%
def gen_triplet_idx(args, fold,loop_i):
    # pair_trP, pair_trN, _, _ = get_pair(args, fold)
    pair_trP, pair_trN = get_pair(args, fold, loop_i)
    np.random.seed(2022)
    triplets = []
    for dis_j in range(args.di_num):
        mir_link_dis_j = pair_trP['mir'][pair_trP['dis'] == dis_j]  
        mir_nolink_dis_j = pair_trN['mir'][pair_trN['dis'] == dis_j] 
        for mir_link in mir_link_dis_j:
            np.random.shuffle(mir_nolink_dis_j)  

            if (args.tile == -1) or (len(mir_nolink_dis_j) < args.tile):
                args.tile = len(mir_nolink_dis_j)
            for mir_nolink in mir_nolink_dis_j[:args.tile]: 
                triplets.append((dis_j, mir_link, mir_nolink))

    pickle.dump({'triplets': triplets}, 
                open(args.fi_out + "Triplet_sample_train/" + "L" + str(loop_i) +"_From_tripletnet2_fold" + str(fold) + ".pkl","wb"))
    print('triplets.shape',len(triplets)) 
    return triplets

# %%
def miNET(args):
    net = keras_models.Sequential()
    # net.add(keras_layers.Dense(128))
    net.add(keras_layers.Dense(256))
    net.add(keras_layers.Dense(args.miemb_size))
    # net.add(keras_layers.Lambda(lambda x: l2_normalize(x, axis=1)))
    return net

def diNET(args):
    net = keras_models.Sequential()
    # net.add(keras_layers.Dense(128))
    net.add(keras_layers.Dense(256))
    net.add(keras_layers.Dense(args.diemb_size))
    # net.add(keras_layers.Lambda(lambda x: l2_normalize(x, axis=1)))
    return net

def tripletNET_2(args):
    in1 = keras_layers.Input(args.didim)
    in2 = keras_layers.Input(args.midim)
    in3 = keras_layers.Input(args.midim)

    f_di = diNET(args)
    f_mi1 = miNET(args)
    f_mi2 = miNET(args)

    di = f_di(in1)
    mi1 = f_mi1(in2)
    mi2 = f_mi1(in3)

    out = keras_layers.Concatenate()([di, mi1, mi2])

    final = keras_models.Model(inputs=[in1, in2, in3], outputs=out)
    return final

# %%
def xuly(args,triplets,fold,loop_i):
    # (1)
    if (args.fi_feature == '../IN/Q18/kfold/') or (args.fi_feature == '../IN/Q18_HMDD3/kfold/'):
        misim_data = pd.read_csv(args.fi_feature + 'L' + str(loop_i) + '/SR_FS' + str(fold+1) + '.csv', header=None)
        disim_data = pd.read_csv(args.fi_feature + 'L' + str(loop_i) + '/SD_SS' + str(fold+1) + '.csv', header=None)
    else:
        misim_data = pd.read_csv(args.fi_feature + 'L' + str(loop_i) + '/SM_fold' + str(fold) + '.csv', header=None)
        disim_data = pd.read_csv(args.fi_feature + 'L' + str(loop_i) + '/SD_fold' + str(fold) + '.csv', header=None)
    idx = np.arange(len(triplets))
    np.random.seed(2022) 
    np.random.shuffle(idx)

    triplets = np.array(triplets)
    tr_di = disim_data.iloc[triplets[idx, 0]]
    tr_mi1 = misim_data.iloc[triplets[idx, 1]]
    tr_mi2 = misim_data.iloc[triplets[idx, 2]]
    print('tr_di.shape, tr_mi1.shape, tr_mi2.shape',tr_di.shape, tr_mi1.shape, tr_mi2.shape)

    # (2) #Train tripletNET_2
    print("\n--- Train tripletNET_2 ...")
    y = np.array([0] * len(triplets)) 
    tripletnet2 = tripletNET_2(args)
    tripletnet2.compile(loss=triplet_loss2, optimizer='adam')
    _ = tripletnet2.fit([tr_di, tr_mi1, tr_mi2], y, epochs=args.epochs, verbose=2)
    print("Done")

    # (3) #TEST TRIPLETNET NOT USE XGBOOST
    te_triplets, te_y = get_sample_triplet2(args, fold, [3, 4], loop_i)
    # idx = np.arange(len(te_triplets)) 
    te_triplets = np.array(te_triplets)
    te_di = disim_data.iloc[te_triplets[:, 0]]
    te_mi1 = misim_data.iloc[te_triplets[:, 1]]
    te_mi2 = misim_data.iloc[te_triplets[:, 2]]

    te_distance = tripletnet2.predict([te_di, te_mi1, te_mi2])

    anchor = te_distance[:, :args.diemb_size]
    positive = te_distance[:, args.diemb_size:args.diemb_size + args.miemb_size]
    negative = te_distance[:, args.diemb_size + args.miemb_size:]

    # -- Save
    te_X = np.concatenate([anchor, positive], axis=1)
    pickle.dump([te_X, te_y], open(args.fi_out + "Data_test/" + "L" + str(loop_i) +"_From_tripletnet2_fold" + str(fold) + ".pkl", "wb"))

    # (4) #GET TRAIN SET FOR TRADITIONAL
    # print("\n\n--- Lay Train cho traditional, tripletnet2")
    tr_triplets, tr_y = get_sample_triplet2(args, fold, [0, 1], loop_i)
    # print(len(tr_triplets))
    # print(len(tr_y))

    idx = np.arange(len(tr_triplets))
    tr_triplets = np.array(tr_triplets)
    tr_di = disim_data.iloc[tr_triplets[idx, 0]]
    tr_mi1 = misim_data.iloc[tr_triplets[idx, 1]]
    tr_mi2 = misim_data.iloc[tr_triplets[idx, 2]]

    tr_distance = tripletnet2.predict([tr_di, tr_mi1, tr_mi2])

    tr_anchor = tr_distance[:, :args.diemb_size]
    tr_positive = tr_distance[:, args.diemb_size:args.diemb_size + args.miemb_size]
    # tr_negative = tr_distance[:, args.diemb_size + args.miemb_size:]  # Chưa dò lại, không dùng cho traditional

    # --- Save
    tr_X = np.concatenate([tr_anchor, tr_positive], axis=1)
    pickle.dump([tr_X, tr_y], open(args.fi_out + "For combination/" + "L" + str(loop_i) +"_Data_train_from_tripletnet2_fold"+ str(fold) + ".pkl", "wb"))
    return

# %%
def main():
    for loop_i in range (1, args.nloop+1):
        print('Loop ', loop_i)
        for fold in range(args.bgf, args.nfold + args.bgf):
            print('fold ',fold)
            triplets = gen_triplet_idx(args, fold, loop_i)
            xuly(args, triplets, fold, loop_i)
    return

if __name__ == "__main__":
    bg_time = time.time()
    main()
    print("Running time: ", time.time() - bg_time, "s")
