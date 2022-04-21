



from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from ssapp.data.Metrics import relRMSE

class PCA_GP_Model():
    def __init__(self, num_components = 10):
        self.gp = GaussianProcessRegressor()
        self.pca = PCA(n_components=num_components)

    def fit(self,X,Y):
        Z = self.pca.fit_transform(Y.reshape(len(X),-1))
        self.org_field_shape = Y.shape
        self.gp.fit(X,Z)


    def transform(self,Y):
        return self.pca.transform(Y)

    def inverse_transform(self,Z):
        return self.pca.inverse_transform(Z)

    def predict(self,X):
        
        Z = self.gp.predict(X)
        return self.pca.inverse_transform(Z).reshape(-1,361,3,4)

    def score(self,X,Y_target,criterion = relRMSE):
        pred = self.predict(X)
        return criterion(pred.flatten(), Y_target.flatten())


