import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

class DATA(object):
    def __init__(self, path_data="C:\\Users\\LEGION\\Desktop\\科研\\读论文\\论文代码\\CEVAE-master\\datasets\\mydataset\\", replications=10):
        self.path_data = path_data
        self.replications = replications
        # which features are binary
        self.binfeats = []
        X = [
            '车相关', '商务住宅', '生活服务', '购物服务', '餐饮服务',
            '科教文化服务', '公共设施', '政府机构及社会团体', '体育休闲服务',
            '医疗保健服务', '风景名胜', '金融保险服务', '住宿服务', 'pop'            
        ]
        # which features are continuous
        self.contfeats = list(range(len(X)))  # Correct way to add 22 to the list

    def __iter__(self):
        for i in range(self.replications):
            # Load data
            df = pd.read_csv(self.path_data + 'df3.csv', encoding='utf-8')
            X = [
            '车相关', '商务住宅', '生活服务', '购物服务', '餐饮服务',
            '科教文化服务', '公共设施', '政府机构及社会团体', '体育休闲服务',
            '医疗保健服务', '风景名胜', '金融保险服务', '住宿服务', 'pop'            
        ]
            x = df[X].values.astype(np.int64)
            t = df[['treat']].values.astype(np.int64)
            y= df[['cos_abs']].values.astype(np.float64)


            # Extracting features from data
            y_cf =  y
            mu_0, mu_1 = y, y


            # Yield the dataset in a tuple form
            yield (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in range(self.replications):
            # Load data
            df = pd.read_csv(self.path_data + 'df3.csv', encoding='utf-8')
            X = [
            '车相关', '商务住宅', '生活服务', '购物服务', '餐饮服务',
            '科教文化服务', '公共设施', '政府机构及社会团体', '体育休闲服务',
            '医疗保健服务', '风景名胜', '金融保险服务', '住宿服务', 'pop'            
        ]
            x = df[X].values.astype(np.int64)
            t = df[['treat']].values.astype(np.int64)
            y= df[['cos_abs']].values.astype(np.float64)

            # Extracting features from data
            y_cf =  y
            mu_0, mu_1 = y, y


            

            # Split data into train, validation, and test
            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            
            # Create train, valid, and test sets
            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            
            # Yield the data splits along with the feature information
            yield train, valid, test, self.contfeats, self.binfeats