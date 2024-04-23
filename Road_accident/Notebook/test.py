import streamlit as st
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from prediction import get_prediction, ordinal_encoder