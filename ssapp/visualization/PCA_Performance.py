from matplotlib import pyplot as plt
import numpy as np
import torch
import seaborn as sns

import sklearn

sns.set_theme()




def plotAEvsPCA(AEmodel,train_fields, test_fields,pca_num = 2,idx = None):

    if idx is None:
        idx = np.random.randint(0,len(test_fields))


    pca = PCA(n_components=pca_num)
    pca.fit_transform(train_fields.reshape((len(train_fields),-1)))

    fig,axs = plt.subplots(nrows = 3, ncols = 1,figsize = (10,11))
    axs[0].set_title('True Field')
    axs[0].plot(test_fields[idx,:,:,1])
    axs[0].grid(True)
    axs[1].set_title('Latent-2D Autoencoder Reconstruction')
    axs[1].plot(autoenc_pred_TEST.detach()[0,:,:,1])
    axs[1].grid(True)
    axs[2].set_title('PCA-'+str(pca_num)+'D Reconstruction')
    axs[2].plot(pca_encode_decode(pca,test_fields[idx:idx+1,::])[0,:,:,1])
    axs[2].grid(True)


def plot_PCA_Loss_vs_num_PC(dataset, ):



def plotEVERYTHING():
    """Plots EVERYTHING"""