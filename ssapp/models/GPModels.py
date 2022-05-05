



from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from ssapp.data.Metrics import relRMSE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class PCA_GP_Model():
    """PCA Gaussian Process Model"""
    def __init__(self, num_components = 10,n_restarts_optimizer=20):
        self.gp = Pipeline([('scaler',StandardScaler()),('Gp',GaussianProcessRegressor(n_restarts_optimizer=n_restarts_optimizer, copy_X_train=False))])
        self.pca = PCA(n_components=num_components)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self,X,Y):
        Z = self.pca.fit_transform(Y.reshape(len(X),-1))
        self.org_field_shape = Y.shape
        self.gp.fit(X,Z)
        self.scaler.fit(X)
        self.is_fitted = True


    def transform(self,Y):
        assert self.is_fitted == True, "fit() has to be called before transformation"
        return self.pca.transform(Y)

    def inverse_transform(self,Z):
        assert self.is_fitted == True, "fit() has to be called before inverse transformation"
        return self.pca.inverse_transform(Z)

    def predict(self,X):
        assert self.is_fitted == True, "fit() has to be called before prediction"
        Z = self.gp.predict(X)
        return self.pca.inverse_transform(Z).reshape(-1,361,3,4)

    def pred_into_latent(self,X):
        return self.gp.predict(X)

    def score(self,X,Y_target,criterion = relRMSE):
        assert self.is_fitted == True
        pred = self.predict(X)
        return criterion(pred.flatten(), Y_target.flatten())


class PCA_LR_Model():
    """PCA Linear Regression Model"""
    def __init__(self, num_components = 10):
        self.LR = LinearRegression()
        self.pca = PCA(n_components=num_components)

    def fit(self,X,Y):
        Z = self.pca.fit_transform(Y.reshape(len(X),-1))
        self.org_field_shape = Y.shape
        self.LR.fit(X,Z)


    def transform(self,Y):
        return self.pca.transform(Y)

    def inverse_transform(self,Z):
        return self.pca.inverse_transform(Z)

    def predict(self,X):
        
        Z = self.LR.predict(X)
        return self.pca.inverse_transform(Z).reshape(-1,361,3,4)

    def score(self,X,Y_target,criterion = relRMSE):
        pred = self.predict(X)
        return criterion(pred.flatten(), Y_target.flatten())


