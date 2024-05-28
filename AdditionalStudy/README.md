# Additional study : TSD behavior and stability towards noise level

## Introduction

This readme gives more detail results and explanation on our study, in particular  :
    
    1. it gives a graphical comparaison of index and model performance through the use of ROC/PR curve

    2. TSD behavior when confronted to various level of noise for Acceptable, Unacceptable and Synthethic ECG record
    
    3. The TSD evolution of the ECG signal for both healthy and pathological patients

All the results shown here can be reproduce by running the following Notebooks:
    - aggregate_results.ipnyb
    - Attractor_shuffling.ipynb
    - First_trial_test_TSD.ipynb
    - Pathology_TSD.ipynb
    - SNR_ECG_TSD.ipynb

## Part 1 : ROC/PR Curve 

 

![alt text](Images/Figure10.png "Figure 1: ROC and PR curves and AUC value (for each curve) for all indexes tested as well as the TSD based SQA method. These AUC curves are obtained using our label convention (i.e., unacceptable ECG are the class 1 we want to predict). The AUC value is given in the legend. The black dashed line corresponds to the case where the model has no class separation capacity (with AUC = 0.5)")

One can observe that "r\_interlead" and HR are the indexes of the set that performs very well.  Both share very high precision (93% for both), recall (63% and 66% respectively), F1-score (75% and 77%) and MCC (72% and 73%). These are the highest scores compare to the other features considered. Differences can be seen, however, in their AUC ROC and PR. In this case, “Norm SNR\_ECG  “and “r\_intralead” shows better performance (with an AUC ROC of 88% and 87% and an AUC PR of 78% and 81%). This also visible graphically in Figure 1. On the other hand, TSD has the worst performance of all. This is correlated with previous results, in particular its necessity of having other indexes that certify the presence of ECG dynamic. It has however one of the high precisions among indexes, implying that it can detect correctly unacceptable signals.