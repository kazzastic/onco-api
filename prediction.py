#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:23:54 2021

@author: kazzastic
"""
from io import BytesIO
from types import new_class
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import zipfile
import pickle

filename = 'logisticReg.sav'


class Predict(object):

    def predictLogistic(self, filePath="n81.csv"):
        cancers = {'0': 'Control', '1': 'AML', '2': 'CML', '3': 'MDS', '4': 'MDS/MPN',
                   '5': 'MPN', '6': 'ALL', '9': 'HL', '10': 'NHL', '11': 'MM', '12': 'APML'}
        model = pickle.load(open(filename, 'rb'))

        "Read the data and remove the NULL values by remove that particular row"
        data = pd.read_csv(filePath)
        y_actual = data['Study_Groups'].to_list()
        nan_values = float("NaN")
        data.replace("#NULL!", nan_values, inplace=True)
        data.dropna(subset=['NE_SFL'], inplace=True)
        dropped_cols = ['Sub_groups1', 'Sub_groups2']
        data.drop(dropped_cols, axis='columns', inplace=True)
        print("Data shape after preprocessing: ", str(data.shape))

        chi2_features = ['LY_WY', 'MO_WY', 'NE_WY', 'LY_WX', 'NE_WZ', 'MO_WZ', 'LY_WZ', 'NE_WX', 'MO_WX', 'NE_SSC', 'MO_Y', 'MO_X', 'MCV', 'LY_X', 'LY_Y',
                         'NE_FSC', 'MO_Z', 'LY_Z', 'RDW_SD', 'Lymph', 'NE_SFL', 'PCV', 'Mono', 'MCHC', 'MCH', 'PLT', 'Neut', 'RDW_CV', 'Hb', 'WBC', 'LYMPH_abs', 'RBC']

        new_x = data[chi2_features]
        print("Data shape with chi square predictors: ", str(new_x.shape))
        y_predicted = model.predict(new_x)
        payload = []
        for i in range(len(y_actual)):
            payload.append(
                {"id": i + 1, "established": y_actual[i], "predicted": cancers[str(y_predicted[i])]})
        df = pd.DataFrame(payload)
        report = classification_report(
            df["established"], df["predicted"], output_dict=True)
        return {"predictions": payload, "report": report}

    def generateCSV(self, file):
        data = pd.read_csv(file, low_memory=False)
        columns_added = ["WBC(10^9/L)", "RBC(10^12/L)", "HGB(g/dL)", "MCV(fL)", "MCH(pg)", "MCH(pg)", "PLT(10^9/L)", "NEUT#(10^9/L)", "LYMPH#(10^9/L)", "MONO#(10^9/L)", "EO#(10^9/L)", "BASO#(10^9/L)", "NEUT%(%)", "LYMPH%(%)", "MONO%(%)", "EO%(%)", "BASO%(%)", "IG#(10^9/L)", "IG%(%)"]
        new_csv = data[columns_added].replace(r'^\s*$', np.NaN, regex=True).replace("----", np.NaN)
        new_csv = new_csv.dropna(subset=columns_added)
        shape = new_csv.shape
        comp_opts = dict(method='zip', archive_name='out.csv')
        new_csv.to_csv('out.zip', index=False, compression=comp_opts)

        return {"Selected Cols": shape}
