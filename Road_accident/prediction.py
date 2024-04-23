import joblib
import numpy as np

from xgboost import XGBClassifier

#def ordinal_encoder(input_val, feats):
 #   feat_val = list(1+np.arange(len(feats)))
 #   feat_key = list(feats)
  #  feat_dict = dict(zip(feat_key, feat_val))
  #  value = feat_dict.map(input_val)
   # return value

#def ordinal_encoder(input_val, feats):
    #for feat in feats:
    #feat_val = list(np.arange(df[feats].nunique()))
    #feat_key = list(df[feat].sort_values().unique())
    #feat_dict = dict(zip(feat_key, feat_val))
    #df[feat] = df[feat].map(feat_dict)
    #return df

def ordinal_encoder(input_val, feats):
    #print("Input value:", input_val)
    #print("Feats list:", feats)
    feat_val = list(range(1, len(feats) + 1))  # Generate ordinal values starting from 1
    feat_dict = dict(zip(feats, feat_val))
    if isinstance(input_val, str):  # Check if input is a single string
        #print("Single input value")
        return feat_dict[input_val]  # Return the ordinal value for the input string
    else:  # If input is a list of strings
        #print("List of input values")
        return [feat_dict[val] for val in input_val]

###
#def ordinal_encoder(input_val, feats):
    #feat_val = list(1+np.arange(len(feats)))
    #feat_key = list(feats)
    #feat_dict = dict(zip(feats, feat_val))
    #value=[feat_dict[val] for val in input_val]
    #return value
"""
def ordinal_encoder(input_val, feats):
    feat_val = list(range(1, len(feats) + 1))  # Generate ordinal values starting from 1
    feat_dict = dict(zip(feats, feat_val))
    if isinstance(input_val, str):  # Check if input is a single string
        return feat_dict[input_val]  # Return the ordinal value for the input string
    else:  # If input is a list of strings
        return [feat_dict[val] for val in input_val]
"""
def get_prediction(data, model):
    """
    Predict the class
    """
    return model.predict(data)

