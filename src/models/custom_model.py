import sys
import numpy as np
import os
import pickle
import pandas as pd
import warnings
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), ".."))

warnings.simplefilter(action="ignore", category=FutureWarning)

folder_model_path = "/workspaces/maitrise/results/Models"
save_path = "/workspaces/maitrise/results"
feat_SQA = ["TSD", "Corr_interlead", "HR", "SNRECG"]
feat_SQANOTSD = ["Corr_interlead", "HR", "SNRECG"]
feat_L2Reg = ["Corr_interlead", "Corr_intralead", "wPMF", "SNRECG", "HR"]
feat_JMI_MI = ["Corr_interlead", "HR", "SNRECG", "Corr_intralead"]

name_SQA_opp_model = "Logit_bin_TSD_Corr_interlead_HR_SNRECG_inverselabel"
name_SQA_model = "Logit_bin_TSD_Corr_interlead_HR_SNRECG_"
name_NTSDSQA_opp_model = "Logit_bin_Corr_interlead_HR_SNRECG_inverselabel"
name_NTSDSQA_model = "Logit_bin_Corr_interlead_HR_SNRECG_"
name_L2_model = "Logit_bin_Corr_interlead_Corr_intralead_wPMF_SNRECG_HR_"
name_L2_opp_model = (
    "Logit_bin_Corr_interlead_Corr_intralead_wPMF_SNRECG_HR_inverselabel"
)
name_JMI_MI_opp_model = "Logit_bin_Corr_interlead_HR_SNRECG_Corr_intralead_inverselabel"
name_JMI_MI_model = "Logit_bin_Corr_interlead_HR_SNRECG_Corr_intralead_"


def sqa_method_score(signals, fs, **kwargs):

    if not os.path.exists(folder_model_path):
        os.mkdir(folder_model_path)
        raise AttributeError("Please, have your model already trained and ready to go!")

    if kwargs.get("opposite"):
        if kwargs["opposite"]:
            name = name_SQA_opp_model
        else:
            name = name_SQA_model
    else:
        name = name_SQA_model

    model = pickle.load(open(os.path.join(folder_model_path, name + ".sav"), "rb"))
    X_test = np.empty(len(feat_SQA))
    for count, name in enumerate(feat_SQA):
        if name == "HR":
            X_test[count] = np.min(method_registry[name](signals, fs))
        else:
            X_test[count] = np.mean(
                method_registry[name](signals, fs, normalization=True)
            )
    X_test = X_test.reshape(1, -1)
    y_proba = model.predict_proba(X_test)
    return y_proba[:, 1]


def sqa_method_lead_score(signals, fs, **kwargs):

    if not os.path.exists(folder_model_path):
        os.mkdir(folder_model_path)
        raise AttributeError("Please, have your model already trained and ready to go!")

    if kwargs.get("opposite"):
        if kwargs["opposite"]:
            name = name_SQA_opp_model
        else:
            name = name_SQA_model
    else:
        name = name_SQA_model

    model = pickle.load(open(os.path.join(folder_model_path, name + ".sav"), "rb"))
    X_test = np.empty([signals.shape[0], len(feat_SQA)])
    for count, name in enumerate(feat_SQA):
        if name == "HR":
            X_test[:, count] = method_registry[name](signals, fs)
        else:
            X_test[:, count] = method_registry[name](signals, fs, normalization=True)
    y_probas = np.empty([signals.shape[0], 2])
    for j in range(len(y_probas)):
        X_test_t = X_test[j, :].reshape(1, -1)
        y_probas[j, :] = model.predict_proba(X_test_t)
    return y_probas[:, 1]


def sqa_ntsd_method_score(signals, fs, **kwargs):
    if not os.path.exists(folder_model_path):
        os.mkdir(folder_model_path)
        raise AttributeError("Please have your model trained and saved!")

    if kwargs.get("opposite"):
        if kwargs["opposite"]:
            name = name_NTSDSQA_opp_model
        else:
            name = name_NTSDSQA_model
    else:
        name = name_NTSDSQA_model
    model = pickle.load(open(os.path.join(folder_model_path, name + ".sav"), "rb"))
    X_test = np.empty([len(feat_SQANOTSD)])
    for count, name in enumerate(feat_SQANOTSD):
        if name == "HR":
            X_test[count] = np.min(method_registry[name](signals, fs))
        else:
            X_test[count] = np.mean(
                method_registry[name](signals, fs, normalization=True)
            )
    X_test = X_test.reshape(1, -1)
    y_proba = model.predict_proba(X_test)
    return y_proba[:, 1]


def model_regularization(signals, fs, **kwargs):

    if not os.path.exists(folder_model_path):
        os.mkdir(folder_model_path)
        raise AttributeError("Please have your model trained and saved!")

    if kwargs.get("opposite"):
        if kwargs["opposite"]:
            name = name_L2_opp_model
        else:
            name = name_L2_model
    else:
        name = name_L2_model

    model = pickle.load(open(os.path.join(folder_model_path, name + ".sav"), "rb"))
    X_test = np.empty([len(feat_L2Reg)])
    for count, name in enumerate(feat_L2Reg):
        if name == "HR":
            X_test[count] = np.min(method_registry[name](signals, fs))
        else:
            X_test[count] = np.mean(
                method_registry[name](signals, fs, normalization=True)
            )
    X_test = X_test.reshape(1, -1)
    y_proba = model.predict_proba(X_test)
    return y_proba[:, 1]


def model_mi(signals, fs, **kwargs):

    if not os.path.exists(folder_model_path):
        os.mkdir(folder_model_path)
        raise AttributeError("Please have your model trained and saved!")

    if kwargs.get("opposite"):
        if kwargs["opposite"]:
            name = name_JMI_MI_opp_model
        else:
            name = name_JMI_MI_model
    else:
        name = name_JMI_MI_model

    model = pickle.load(open(os.path.join(folder_model_path, name + ".sav"), "rb"))
    X_test = np.empty([len(feat_JMI_MI)])
    for count, name in enumerate(feat_JMI_MI):
        if name == "HR":
            X_test[count] = np.min(method_registry[name](signals, fs))
        else:
            X_test[count] = np.mean(
                method_registry[name](signals, fs, normalization=True)
            )
    X_test = X_test.reshape(1, -1)
    y_proba = model.predict_proba(X_test)
    return y_proba[:, 1]
