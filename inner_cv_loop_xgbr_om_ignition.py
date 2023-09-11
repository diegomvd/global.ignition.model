import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score

    
# custom transformer for sklearn pipeline
class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        return X[self.cols]

    def fit(self, X, y=None):
        return self    
    
 
def keep_predictor(plist,pname,pval):
    if pval>0.5:
        plist.append(pname)
    return plist


def build_predictor_list(predictor_dict):
    plist = []
    for predictor in predictor_dict:
        plist = keep_predictor(plist,predictor,predictor_dict[predictor])
    return plist    

   
# Read data from the corresponding outer fold.
fold = 1
data = pd.read_csv("./folds/fire_fold_{}.csv".format(fold))



# Predictors for testing:
# fmc=1

# prec30=1
# prec7=0
# prec=0

# rh=0

# temp30_media=1
# temp30_moda=1
# temp7_media=0
# temp7_moda=1
# temp_max=0

# wind_max=1

# lm14=1
# lm7=0
# ls14=0
# ls7=0

# aspect=0
# elev=0
# rough=0
# slope=0
# road=0

# pop_density=0
# livestock = 0

# ecorregion = 0

# cci = 0 
# real = 0

# seed = 1

# md = 2
# e=0.2
# mcw=1
# subsample=0.7
# g=5
# mds = 1


# Dictionary of predictors.
predictor_dict = {
    "FMC": fmc,
    "Prec30" : prec30,
    "Prec7" : prec7,
    "Prec" : prec,
    "Rh" : rh,
    "Temp30_media": temp30_media,
    "Temp30_moda" : temp30_moda,
    "Temp7_media": temp7_media,
    "Temp7_moda": temp7_moda,
    "Temp_max" : temp_max,
    "wind_max" : wind_max,
    "lm14": lm14,
    "lm7": lm7,
    "ls14" : ls14,
    "ls7" : ls7,
    "aspect": aspect,
    "elev": elev,
    "rough": rough,
    "slope": slope,
    "road": road,
    "pop_density": pop_density,
    "livestock" : livestock ,
    "ecorregion" : ecorregion ,
    "cci": cci , 
    "real": real ,
}


if seed<0:
    seed = -1*seed

# Extract the target and predictors.
predictor_list = build_predictor_list(predictor_dict)
npredictors = len(predictor_list)

if npredictors > 0:

    n_splits = 5
    cvs = np.zeros(n_splits)
    
    mdint = int(md)
    
    cls = XGBClassifier(
            n_estimators=100,
            learning_rate=e,
            max_depth = mdint,
            min_child_weight=mcw,
            subsample=subsample,
            min_split_loss = g,
            max_delta_step = mds,
            random_state = seed
            )
    

    estimator = Pipeline([
        ("col_extract", ColumnExtractor(predictor_list)),
        ("regressor",cls)
    ])
        
    # print("Starting.")
     
    # Set up 5 folds for inner cross-validation with 1 repeat because the evolutionary algorithm will resample anyway.
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=1,random_state=seed)

    y = np.array(data["ign"])
    X = data.drop("ign",axis="columns")

    # Using as criterion mean squared error for a predictor in the log space is equivalent to minimize the mean absolute error in the natural space. Thus, at each tree-node split the algorithm is minimizing the mean absolute error of biomass density.
    cvs = cross_val_score(estimator,X,y,cv=rkf,scoring="accuracy")    
    # cvs = 0
    # estimator.fit(X,y)

    # Outputs for multi-objective optimization with NSGA-II with OpenMole.
    accuracy = np.mean(cvs)*100
    print(accuracy)

else:
    # Predicted value is the majority class. A simple way to find it is to calculate mean and see if it closer to 0 or 1. 
    y = np.array(data["ign"])
    pred = np.int(np.mean(y))
    accuracy = accuracy_score(y,pred)*100
    # print(error)

    




