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


def plot3DPCA(dataset, 
                pca_components = (1,2,3),
                param = 0,
                dataset_name = None,
                view_init=(55,0),
                figsize = (8,8)):
    assert len(pca_components) == 3
    assert [x > 0 for x in pca_components]
    pca_components = [x-1 for x in pca_components] # Switch to zero-index
    num_samples = len(dataset)
    
    dataloader = DataLoader(dataset,batch_size=num_samples)
    params, fields  = next(iter(dataloader))

    pca = sklearn.decomposition.PCA(n_components = max(pca_components)+1)

    pca_results = pca.fit_transform(fields.reshape(num_samples,-1))[:,list(pca_components)]
    
    p1,p2,p3 = pca_components

    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(projection='3d')
    #ax.view_init(view_init[0],view_init[1])

    ax.scatter(pca_results[:,p1],pca_results[:,p2],pca_results[:,p3],c = params[:,param],cmap = 'plasma',depthshade=True)
    



def plot3DContour(dataset, 
                pca_components = (1,2,3),
                param = 0,
                dataset_name = None,
                view_init=(55,0),
                proj_scalers = (2.5,2.5,2.5)):
    assert len(pca_components) == 3
    dataset_name = dataset.name
    pca_components = [x-1 for x in pca_components] # Switch to zero-index
    num_samples = len(dataset)
    sns.set_theme(style="whitegrid")
    size = 10
    
    p1,p2,p3 = pca_components

    dataloader = DataLoader(dataset,batch_size=num_samples)
    params, fields  = next(iter(dataloader))

    pca = sklearn.decomposition.PCA(n_components = max(pca_components)+1)

    pca_results = pca.fit_transform(fields.reshape(num_samples,-1))[:,pca_components]
    
    


    X,Y,Z = (pca_results[:,0],pca_results[:,1],pca_results[:,2])

    X_max,Y_max,Z_max = (min(X),max(Y),min(Z))

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X, Y, Z,c = params[:,param],cmap = 'plasma',depthshade=True,s=size)

    fig.suptitle('3D point cloud of '+dataset_name+' unto 3 main principle components')
    cset = ax.scatter(np.ones_like(X)*X_max*proj_scalers[0]-0.01*X, Y, Z,c = params[:,param],cmap = 'plasma',depthshade=True,s=size)

    cset = ax.scatter(X, np.ones_like(Y)*Y_max*proj_scalers[1]-0.01*Y, Z,c = params[:,param],cmap = 'plasma',depthshade=True,s=size)
    cset = ax.scatter(X, Y, np.ones_like(Z)*Z_max*proj_scalers[2]-0.01*Z,c = params[:,param],cmap = 'plasma',depthshade=True,s=size)
    ax.set_xlabel('PCA'+str(pca_components[0]+1))
    ax.set_ylabel('PCA'+str(pca_components[1]+1))
    ax.set_zlabel('PCA'+str(pca_components[2]+1))
    


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
                                    pca_components = [1,2],
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
            elif component == 'cross':
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


def plotFieldComparison(Y_pred,
                        Y_truth,
                        idx = 0,
                        component = 'co',
                        phis = [0],
                        transform = lambda a,b : 20*np.log10(a**2+b**2),
                        title = 'Field Prediction and Truth'):

    assert component == 'co' or component == 'cross'
    assert len(Y_pred) == len(Y_truth)

    if component == 'co':
        X_plot_field = transform(Y_pred[idx,:,phis,0],Y_pred[idx,:,phis,1])
        Y_plot_field = transform(Y_truth[idx,:,phis,0],Y_truth[idx,:,phis,1])
    elif component == 'cross':
        X_plot_field = transform(Y_pred[idx,:,phis,2],Y_pred[idx,:,phis,3])
        Y_plot_field = transform(Y_truth[idx,:,phis,2],Y_truth[idx,:,phis,3])

    thetas =  np.array([np.linspace(-180,180,361) for x in phis]).T
    print(thetas.shape)
    plt.figure()
    plt.title(title)
    plt.plot(thetas,X_plot_field,label='Pred.')
    plt.plot(thetas,Y_plot_field.T,label='Truth') # WHY DO I NEED THE TRANSPOSE????
    plt.legend()
    plt.xlabel(r'$\theta\degree$')

    
    if component == 'co':
        plt.ylabel('$|E_{co}|$ dB')
    else:
        plt.ylabel('$|E_{cross}|$ dB')

def closestIdxFromParams(dataset,params):

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    dataloader = DataLoader(dataset,batch_size = len(dataset))
    params, fields = next(iter(dataloader))

    assert len(params) == len(params)
    
    return find_nearest()


def plotGPvsPCADimensions(dataset, max_number_pca = 20):
    from ssapp.Utils import train_test_data_split
    from ssapp.data.Metrics import relRMSE
    from sklearn.decomposition import PCA
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler


    train_data, test_data = train_test_data_split(dataset)

    train_loader = DataLoader(train_data, batch_size = len(train_data))
    test_loader = DataLoader(test_data,batch_size = len(test_data))

    TRAIN_PARAMS,TRAIN_FIELDS = next(iter(train_loader))
    TEST_PARAMS,TEST_FIELDS = next(iter(test_loader))


    Number_PCAs = []

    sweep_info = {
            'PCR Train Rec. Loss': [],
            'PCR Validation Rec. Loss': [],
            'GP Train Latent Loss': [],
            'GP Validation Latent Loss': [],
            'GP-PCA Train Rec. Loss': [],
            'GP-PCA Validation Rec. Loss': [],
            'LR-PCA Train Rec. Loss': [],
            'LR-PCA Validation Rec. Loss': [],
            '2D AE Train Rec. Loss': [],
            '2D AE Validation Rec. Loss' : [],
            'GP-AE Train Rec. Loss': [],
            'GP-AE Validation Rec. Loss': []
            }

    # Make direct prediction baseline

    gpr = Pipeline([('scaler', StandardScaler()), ('gp', GaussianProcessRegressor())]).fit(TRAIN_PARAMS,TRAIN_FIELDS.reshape(len(TRAIN_FIELDS),-1))
    train_baseline = relRMSE(TRAIN_FIELDS.flatten(), gpr.predict(TRAIN_PARAMS).flatten())
    test_baseline = relRMSE(TEST_FIELDS.flatten(), gpr.predict(TEST_PARAMS).flatten())

    # Now with varying size latent models

    for num_pca in range(1,max_number_pca+1):

        Number_PCAs.append(num_pca)
        pca = PCA(n_components=num_pca)
        pca_train = pca.fit_transform(TRAIN_FIELDS.reshape((len(train_data),-1)))
        pca_val = pca.transform(TEST_FIELDS.reshape((len(TEST_FIELDS),-1)))
        PCA_TRAIN_RECONSTRUCTED_FIELD = pca.inverse_transform(pca_train).reshape(len(TRAIN_FIELDS),361,3,4)
        PCA_TEST_RECONSTRUCTED_FIELD = pca.inverse_transform(pca_val).reshape(len(TEST_FIELDS),361,3,4)

        gpr = Pipeline([('scaler', StandardScaler()), ('gp', GaussianProcessRegressor())]).fit(TRAIN_PARAMS, pca_train)
        #gpr = GaussianProcessRegressor().fit(TRAIN_PARAMS, pca_train)

        sweep_info['GP Train Latent Loss'].append(relRMSE(gpr.predict(TRAIN_PARAMS), pca_train))
        sweep_info['GP Validation Latent Loss'].append(relRMSE(gpr.predict(TEST_PARAMS),pca_val))

        sweep_info['PCR Train Rec. Loss'].append(relRMSE(TRAIN_FIELDS.flatten(), PCA_TRAIN_RECONSTRUCTED_FIELD.flatten()))
        sweep_info['PCR Validation Rec. Loss'].append(relRMSE(TEST_FIELDS.flatten(), PCA_TEST_RECONSTRUCTED_FIELD.flatten()))

        ## Loss in reconstruction
        GPR_TRAIN_RECONSTRUCTED_FIELD = pca.inverse_transform(gpr.predict(TRAIN_PARAMS)).reshape(len(TRAIN_PARAMS),361,3,4)
        GPR_TEST_RECONSTRUCTED_FIELD = pca.inverse_transform(gpr.predict(TEST_PARAMS)).reshape(len(TEST_PARAMS),361,3,4)

        sweep_info['GP-PCA Train Rec. Loss'].append(relRMSE(TRAIN_FIELDS.flatten(), GPR_TRAIN_RECONSTRUCTED_FIELD.flatten()))

        sweep_info['GP-PCA Validation Rec. Loss'].append(relRMSE(TEST_FIELDS.flatten(), GPR_TEST_RECONSTRUCTED_FIELD.flatten()))


    MEDIUM_LINEWIDTH = 2
    SMALL_LINEWIDTH = 1

    plt.figure(figsize = (8,4))
    plt.semilogy(Number_PCAs,sweep_info['PCR Train Rec. Loss'],label ='PCR Train Rec. Loss',linewidth = MEDIUM_LINEWIDTH)
    plt.semilogy(Number_PCAs,sweep_info['PCR Validation Rec. Loss'],label='PCR Validation Rec. Loss',linewidth = MEDIUM_LINEWIDTH)
    plt.semilogy(Number_PCAs,sweep_info['GP-PCA Train Rec. Loss'],label = 'GP-PCA Train Rec. Loss',linewidth = MEDIUM_LINEWIDTH)
    plt.semilogy(Number_PCAs,sweep_info['GP-PCA Validation Rec. Loss'],label = 'GP-PCA Validation Rec. Loss',linewidth = MEDIUM_LINEWIDTH)
    plt.axhline(train_baseline,label = 'GP Baseline Train Rec. Loss',linestyle = '--',c = 'green',linewidth = SMALL_LINEWIDTH)
    plt.axhline(test_baseline,label = 'GP Baseline Validation Rec. Loss',linestyle = '--',c = 'red',linewidth = SMALL_LINEWIDTH)
    plt.xlim(1,num_pca)
    plt.xticks([x for x in range(1,max_number_pca+1)])
    plt.legend()
    plt.xlabel('Latent Space Dimensions')
    plt.ylabel('relRMSE Reconstruction Loss')
    plt.title('PCA and Latent-Regression Reconstruction Loss')
    plt.grid(True)

