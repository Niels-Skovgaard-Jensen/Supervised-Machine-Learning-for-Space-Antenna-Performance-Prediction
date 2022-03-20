from matplotlib import pyplot as plt
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from ssapp.data.AntennaDatasetLoaders import PatchAntennaDataset
from torch.utils.data.dataloader import DataLoader
from ssapp.data.AntennaDatasetLoaders import PatchAntennaDataset
import ssapp.Utils as Utils


def plotEncodingVerificaiton(model, train_field):

    EXTENT = [0,90,-180,180]
    ASPECT = 0.3
    LINEWIDTH = 3
    X_TEXT_DISPLACEMENT = -45
    thetas = np.linspace(-180,180,361)
    def setImgAxis(ax):
        ax.set_xlabel(r'$ \phi $')
        ax.set_ylabel(r'$ \theta $')
        ax.set_aspect(ASPECT) 

    fig,axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axs[0,0].imshow(torch.abs(train_field[0,:,:,1:2]),aspect = 1/100, interpolation = 'none',extent=EXTENT)
    axs[0,0].set_title('Truth')
    axs[0,0].text(X_TEXT_DISPLACEMENT,0,'Co-polar',rotation = 'vertical',fontsize = 13, va = 'center')
    setImgAxis(axs[0,0])

    axs[0,1].imshow(torch.abs(model(train_field.float())[0,:,:,1:2]).detach(), interpolation = 'none',extent=EXTENT)
    axs[0,1].set_title('Decoded')
    axs[0,1].axvline(x = 45, color = 'r', linestyle = '--')
    setImgAxis(axs[0,1])

    axs[0,2].plot(thetas,10*torch.log10(torch.abs(train_field[0,:,1,1:2])),label = 'Data',linewidth = LINEWIDTH)
    axs[0,2].plot(thetas,10*torch.log10(torch.abs(model(train_field.float())[0,:,1,1:2])).detach(),label = 'Model',linewidth = LINEWIDTH) 
    axs[0,2].grid()
    axs[0,2].set_title('Middle-Cut at $\phi = 45$')
    axs[0,2].set_xlabel(r'$\theta$')
    axs[0,2].legend()
    axs[0,2].set_ylabel('Copolar Amplitude [dB]')

    axs[1,0].imshow(torch.abs(train_field[0,:,:,3:4]),aspect = 1/100, interpolation = 'none',extent=EXTENT)
    axs[1,0].text(X_TEXT_DISPLACEMENT,0,'Cross-polar',rotation = 'vertical',fontsize = 13, va = 'center')
    setImgAxis(axs[1,0]) 
    

    axs[1,1].imshow(torch.abs(model(train_field.float())[0,:,:,3:4]).detach(),aspect = 1/100, interpolation = 'none',extent=EXTENT) 
    setImgAxis(axs[1,1])
    axs[1,1].axvline(x = 45, color = 'r', linestyle = '--')

    axs[1,2].plot(thetas,10*torch.log10(torch.abs(train_field[0,:,1,3:4])),linewidth = LINEWIDTH)
    axs[1,2].plot(thetas,10*torch.log10(torch.abs(model(train_field.float())[0,:,1,3:4])).detach(),linewidth = LINEWIDTH) 
    axs[1,2].set_xlabel(r'$\theta$')
    axs[1,2].grid()
    axs[1,2].set_ylabel('Crosspolar Amplitude [dB]')


def plt_2D_PCA()

    BATCH_SIZE = 1

    dataset = PatchAntennaDataset()
    train_data, test_data = Utils.train_test_data_split(dataset, TRAIN_TEST_RATIO = 0.7)

    train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=True)

    pca = PCA(n_components=10)
    train_loader = DataLoader(train_data,batch_size=bs,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=bs,shuffle=True)
    TRAIN_PARAMS,TRAIN_FIELDS = next(iter(train_loader))
    TEST_PARAMS,TEST_FIELDS = next(iter(train_loader))

    pca_results = pca.fit_transform(TEST_FIELDS.reshape((bs,-1)))
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))

    print(TEST_FIELDS.reshape((len(TEST_FIELDS),-1)).shape)
    print(TEST_PARAMS.reshape((len(TEST_FIELDS),-1)).shape)

    param_names = ['Coax Placement-X','Coax Placement-Y','Substrate Permitivity']

    fig, axs = plt.subplots(nrows = 1, ncols = 3,figsize = (14,3))
    fig.suptitle('Patch Antenna Dataset PCA Projection With Parameter Coloring')
    for i in range(0,3):
        im = axs[i].scatter(pca_results[:,0],pca_results[:,1],c = TEST_PARAMS[:,i],cmap = 'plasma')
        cbar = plt.colorbar(im,ax=axs[i])
        cbar.set_label(param_names[i])