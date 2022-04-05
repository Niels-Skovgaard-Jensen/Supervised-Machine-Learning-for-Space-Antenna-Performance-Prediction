from matplotlib import pyplot as plt
import numpy as np
import torch
import seaborn as sns
from torch.utils.data.dataloader import DataLoader
from sklearn.decomposition import PCA
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

def plotInverseTransformStandardPCA(dataset,
                                    plot_latent_space = False,
                                    num_std_dev = 1,
                                    phi_cut = [0],
                                    phi_labels = ['0'],
                                    num_rows = 3,
                                    num_cols = 3,
                                    component = 'co',
                                    transform = lambda a,b : 20*np.log10(a**2+b**2),
                                    ylabel = 'dB'):

    assert component == 'co' or component == 'cross'
    assert len(phi_cut) == len(phi_labels)



    ylabel = r'$|E_{'+component+'}|$ '+ylabel

    dataloader = DataLoader(dataset,batch_size = len(dataset))
    params, fields = next(iter(dataloader))

    

    pca = PCA(n_components=2).fit(fields.reshape(len(fields),-1))

    latent_points = pca.transform(fields.reshape(len(fields),-1))


    latent_std_dev = np.std(latent_points,axis= 0)
    latent_mean = np.mean(latent_points, axis = 0)

    X = np.linspace(latent_mean[0]-num_std_dev*latent_std_dev[0],latent_mean[0]+num_std_dev*latent_std_dev[0],num_cols)
    Y = np.linspace(latent_mean[1]-num_std_dev*latent_std_dev[1],latent_mean[1]+num_std_dev*latent_std_dev[1],num_rows)
    X,Y = np.meshgrid(X,Y)
    theta = np.linspace(-180,180,len(fields[0,:,0,0]))

    fig = plt.figure(figsize = (8,9.5),constrained_layout=True)
    subfigs = fig.subfigures(2, 1, height_ratios=[0.7, 1.])

    ax = subfigs[0].add_subplot(1,1,1, adjustable='box', aspect=1)
    ax.set_title('2D Latent Space')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')

    ax.scatter(latent_points[:,0],latent_points[:,1],marker='o',s=1,label = 'Projected Data')
    ax.scatter(X,Y,marker='x',color = 'red',s = 120,label = 'Reconstructed Point')


    subfigs[0].legend()
    
    axs = subfigs[1].subplots(nrows=num_rows,ncols = num_cols, sharex='all', sharey='row')


    for i,ax in enumerate(axs.flatten()):
        pred = pca.inverse_transform([X.flatten()[i],Y.flatten()[i]]).reshape(1,361,3,4)
        
        for i,phi in enumerate(list(phi_cut)):
            if component == 'co':
                plot_field = transform(pred[0,:,phi,0],pred[0,:,phi,1])
            else:
                plot_field = transform(pred[0,:,phi,2],pred[0,:,phi,3])
            ax.plot(theta,plot_field,label = phi_labels[i])

        ax.set_xlim([-180,180])
    

    for ax in axs[-1,:]:
        ax.set_xlabel(r'$\theta\degree$')
    for ax in axs[:,0]:
        ax.set_ylabel(ylabel)
    handles, labels = subfigs[1].gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    subfigs[1].legend(by_label.values(), by_label.keys(), loc = 'upper center',ncol = len(phi_labels), bbox_to_anchor=(0.5,-0.1))
    
    







def plotEVERYTHING():
    """Plots EVERYTHING"""

