import joblib
import numpy as np

from xgboost import XGBClassifier



def ordinal_encoder(input_val, feats):
    feat_val = list(1+np.arange(len(feats)))
    feat_key = list(feats)
    feat_dict = dict(zip(feats, feat_val))
    value=feat_dict[input_val]
    return value


 
def get_prediction(data, model):
    """
    Predict the class
    """
    return model.predict(data)


 