from GENERAL_UTILS import PlotCurveKFOLD as myplot
from GENERAL_UTILS.CalcScores import calc_scores_KFOLD
import matplotlib.pyplot as plt
import numpy as np

# %%
def gen_setvalue(args, path, cla, ther,b_ix, e_ix, loai_tam):
    arr_y_loai = []
    arr_y_pred = []

    for i in range(b_ix, e_ix):
        if loai_tam == '_fold':
            path_loai_i = path + loai_tam + str(i) + '.csv'
        else:
            path_loai_i = path + loai_tam + str(args.set_dis[i]) + '.csv'
        arr_y_loai_i = np.genfromtxt(path_loai_i)
        arr_y_loai.append(arr_y_loai_i)
        arr_y_pred_i = np.where(arr_y_loai_i >= ther, 1, cla)

        arr_y_pred.append(arr_y_pred_i)

    return arr_y_loai, arr_y_pred

# %%
def eval(args, path_in,b_ix, e_ix,str_fi1,str_fi2,cla,ther,path_out_fig,path_out_kq,name_fig_file,loai_tam):
    arr_y_prob, arr_y_pred = gen_setvalue(args, path_in + str_fi1, cla, ther, b_ix, e_ix, loai_tam=loai_tam)
    arr_y_test, arr_y_pred_temp = gen_setvalue(args, path_in + str_fi2, cla, ther, b_ix, e_ix, loai_tam=loai_tam)
    ############################
    # vẽ Roc và Precision-recall
    if args.type_eval == 'kfold':
        myplot.draw_plot_KFOLD(plt, arr_y_test, arr_y_prob,name_fig_file,loai_tam)
    ############################

    return

# %%
