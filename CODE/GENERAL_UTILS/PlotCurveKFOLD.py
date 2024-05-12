import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from scipy import interp


def draw_plot_KFOLD(plt, arr_y_test, arr_y_prob,name_fig_file,loai_tam):
    if loai_tam=='_kfold':
        temp = 'fold'
    else:
        temp = 'dis'
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11,4))
    ax_roc = ax[0]
    ax_rpc = ax[1]

    n_samples = 0
    for i, y_test in enumerate(arr_y_test):
        n_samples += y_test.shape[0]
    mean_fpr = np.linspace(0, 1, n_samples)
    roc_aucs = []; tprs = []

    mean_rec = np.linspace(0, 1, n_samples)
    pres = []; rpc_aucs = []

    # get fpr, tpr scores
    for i, (y_test, y_prob) in enumerate(zip(arr_y_test, arr_y_prob)):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        # plot ROC curve
        if len(arr_y_test) > 1: #5fold
            ax_roc.plot(fpr, tpr, lw=1, alpha=0.5, label='ROC '+ temp +' %d (AUC = %0.4f)'% (i+1, roc_auc)) #@@@Q
        
        interp_tpr = interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        roc_aucs.append(roc_auc)

    ax_roc.plot([0, 1], [0, 1], lw=2, alpha=.8, linestyle='--', color='r', label='Chance') #@Q
    
    # Ve ROC mean
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_roc_auc = auc(mean_fpr, mean_tpr)
    std__roc_auc = np.std(roc_aucs)
    
    if len(arr_y_test) > 1:
        ax_roc.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_roc_auc, std__roc_auc),
            lw=2, alpha=.8)
    else:
        ax_roc.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.4f)' % (mean_roc_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2) # ,label=r'$\pm$ 1 std. dev.')
    
    # Dat ten 
    ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax_roc.set_title(label="ROC curve", fontweight='bold')
    ax_roc.set_xlabel('False Positive Rate', fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontweight='bold')
    ax_roc.legend( fontsize='x-small',loc='lower right', bbox_to_anchor=(1, 0.05)) #tính từ toạ độ (1,0.05), legend nằm ở lower right

    # get precision, recall scores
                   
    for i, (y_test, y_prob) in enumerate(zip(arr_y_test, arr_y_prob)):
        precision, recall,  _ = precision_recall_curve(y_test, y_prob)
        rpc_aupr = average_precision_score(y_test, y_prob)
        # plot precision recall curve
        # @Q vẽ 5 fold
        if len(arr_y_test) > 1:
            ax_rpc.plot(recall, precision, lw=1, alpha=0.5, label='PR ' + temp + ' %d (AP = %0.4f)'% (i+1, rpc_aupr)) #@@

        interp_pre = interp(mean_rec, recall, precision)
        # interp_tpr[0] = 0.0
        pres.append(interp_pre)
        rpc_aucs.append(rpc_aupr)
    
    y_tests = np.array([])
    for y_test in arr_y_test:
        y_tests = np.hstack((y_tests, y_test.ravel()))
    
    no_skill = len(y_tests[y_tests==1]) / y_tests.shape[0]
    ax_rpc.plot([0, 1], [no_skill, no_skill], linestyle='--', lw=2, color='r', label='Chance')
    
    
    # Ve duong mean
    all_y_test = np.concatenate(arr_y_test)
    all_y_prob = np.concatenate(arr_y_prob)
    precision, recall, _ = precision_recall_curve(all_y_test, all_y_prob)
    
    # --- Lay TB cac fold
    ave = []
    for i in range(len(arr_y_test)):
        ave.append(average_precision_score(arr_y_test[i], arr_y_prob[i]))
    
    
    if len(arr_y_test) > 1:
        std__pr_aupr = np.std(rpc_aucs)
        ax_rpc.plot(recall, precision, color='b',
             label=r'Mean PR (AUPR = %0.4f $\pm$ %0.4f)' %
             (average_precision_score(all_y_test, all_y_prob),std__roc_auc), # Cu
             # (np.mean(ave),np.std(ave)), # moi
             lw=2, alpha=1)
    else:
        ax_rpc.plot(recall, precision, color='b',
             label=r'Mean PR (AUPR = %0.4f)' %
             (average_precision_score(all_y_test, all_y_prob)), # CU
             # (np.mean(ave)), # moi
             lw=2, alpha=1)

 
    # Dat ten
    ax_rpc.set_title('Precision-Recall Curve', fontweight='bold')
    ax_rpc.set_xlabel('Recall', fontweight='bold')
    ax_rpc.set_ylabel('Precision', fontweight='bold')
    ax_rpc.legend(fontsize='x-small',loc='lower left', bbox_to_anchor=(0.05, 0.05))

    # return plt
    
    plt.savefig(name_fig_file +'.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(name_fig_file +'.png',format='png', dpi=300,bbox_inches='tight')
    plt.show()
   