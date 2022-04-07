from matplotlib import pyplot as plt
import numpy as np
import torch
import seaborn as sns
from torch.utils.data.dataloader import DataLoader
from sklearn.decomposition import PCA
import sklearn

sns.set_theme()




def plotParameterColoredLatentSpace(dataset,param_names = None,pca_components = (0,1),figsize = (14,2.75)):

    pca_components = [x-1 for x in pca_components] # Switch to zero-index
    num_samples = len(dataset)
    
    dataloader = DataLoader(dataset,batch_size=num_samples)
    params, fields  = next(iter(dataloader))
    num_params = len(params.T)


    pca = sklearn.decomposition.PCA(n_components = max(pca_components)+1)

    projection = pca.fit_transform(fields.reshape(num_samples,-1))[:,list(pca_components)]


    if param_names is None:
        param_names = ['Parameter '+str(x) for x in range(0,num_params)]
    
    fig, axs = plt.subplots(nrows = 1, ncols = num_params,figsize = figsize)
    fig.suptitle(dataset.name +' PCA Projection With Parameter Coloring', fontsize = 16)
    axs[0].set_ylabel('PCA '+str(pca_components[1]+1))
    for i in range(0,num_params):
        im = axs[i].scatter(projection[:,0],projection[:,1],c = params[:,i],cmap = 'plasma')
        axs[i].set_xlabel('PCA '+str(pca_components[0]+1))
        
        cbar = plt.colorbar(im,ax=axs[i])
        cbar.set_label(param_names[i])
        

def plotPCAVariance(dataset,num_components = 10, dataset_name = None):

    if dataset_name == None:
        dataset_name = dataset.name

    num_samples = len(dataset)
    
    dataloader = DataLoader(dataset,batch_size=num_samples)
    params, fields  = next(iter(dataloader))
    pca = sklearn.decomposition.PCA(n_components = num_components)

    pca.fit(fields.reshape(num_samples,-1))
    plt.figure()
    sns.barplot(x = list(range(1,1+len(pca.explained_variance_ratio_))),y = pca.explained_variance_ratio_*100)
    plt.xlabel('PCA Number')
    plt.ylabel('Variance Explained %')
    plt.title(dataset.name+' Variance Explained by Prinicipal Components')




def pltAspectRatio(x):
    x1 = x[:,0]
    x2 = x[:,1]
    return (max(x1)-min(x1))/(max(x2)-min(x2))

def plotInverseTransformStandardPCA(dataset,
                                    plot_latent_space = False,
                                    num_std_dev = 1,
                                    phi_cut = [0],
                                    phi_labels = None,
                                    num_rows = 3,
                                    num_cols = 3,
                                    component = 'co',
                                    transform = lambda a,b : 20*np.log10(a**2+b**2),
                                    ylabel = 'dB',
                                    pca_components = [0,1],
                                    plot_deviation_from_mean = False,
                                    polar_coordiantes = True):

    assert component == 'co' or component == 'cross'
    assert len(pca_components) == 2

    

    pca_components = [x-1 for x in pca_components] # Switch to zero-index

    ylabel = r'$|E_{'+component+'}|$ '+ylabel

    dataloader = DataLoader(dataset,batch_size = len(dataset))
    params, fields = next(iter(dataloader))
    
    
    pca = PCA(n_components=max(pca_components)+1).fit(fields.reshape(len(fields),-1))

    latent_points = pca.transform(fields.reshape(len(fields),-1))[:,list(pca_components)]

    latent_std_dev = np.std(latent_points,axis= 0)
    latent_mean = np.mean(latent_points, axis = 0)

    X = np.linspace(latent_mean[0]-num_std_dev*latent_std_dev[0],latent_mean[0]+num_std_dev*latent_std_dev[0],num_cols)
    Y = np.linspace(latent_mean[1]-num_std_dev*latent_std_dev[1],latent_mean[1]+num_std_dev*latent_std_dev[1],num_rows)
    X,Y = np.meshgrid(X,Y)
    theta = np.linspace(-180,180,len(fields[0,:,0,0]))

    fig = plt.figure(figsize = (8,9.5),constrained_layout=True)
    subfigs = fig.subfigures(2, 1, height_ratios=[0.7, 1.])

    ax = subfigs[0].add_subplot(1,1,1, adjustable='box', aspect=pltAspectRatio(latent_points))
    ax.set_title('2D Latent Space')
    ax.set_xlabel('PCA '+str(pca_components[0]+1))
    ax.set_ylabel('PCA '+str(pca_components[1]+1))

    ax.scatter(latent_points[:,0],latent_points[:,1],marker='o',s=1,label = 'Projected Data')
    ax.scatter(X,Y,marker='x',color = 'red',s = 120,label = 'Reconstructed Point')

    subfigs[0].legend()


    axs = subfigs[1].subplots(nrows=num_rows,ncols = num_cols, sharex='all', sharey='row')

    latent_pred_point = np.zeros(shape = (max(pca_components)+1))
    
    for i,ax in enumerate(axs.flatten()):
        latent_pred_point[pca_components[0]] = X.flatten()[i]
        latent_pred_point[pca_components[1]] = Y.flatten()[i]
        pred = pca.inverse_transform(latent_pred_point).reshape(1,361,3,4)
        
        temp_max,temp_min = (0,0)
        for i,phi in enumerate(list(phi_cut)):
            if component == 'co':
                plot_field = transform(pred[0,:,phi,0],pred[0,:,phi,1])
            else:
                plot_field = transform(pred[0,:,phi,2],pred[0,:,phi,3])
            ax.plot(theta,plot_field)
            temp_min = min(temp_min,min(plot_field))
            temp_max = max(temp_max,max(plot_field))
        
        ax.set_ylim([max(-150,temp_min),temp_max*1.2])
            
        
        

        ax.set_xlim([-180,180])
        
    

    for ax in axs[-1,:]:
        ax.set_xlabel(r'$\theta\degree$')
    for ax in axs[:,0]:
        ax.set_ylabel(ylabel)







def plotEVERYTHING():
    """Plots EVERYTHING"""

