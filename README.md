# ecg_evaluation
Project to evaluate ECG quality and noise level

## Project Goal :

This repo contains all the indexes and models implemented for ECG signal quality assessment. It was done in the context of this study (cite my master thesis?????).
The project was done in python using **Docker**[[1]](#1) with the docker image [PDM](https://hub.docker.com/r/frostming/pdm)

## Implementations :

### Index :

The indexes implemented are the following :
- **wPMF** [[2]](#2)
- **SNR** [[3]](#3) : This is an adaptation. We add a normalized version of this.
- **TSD** [[4]](#4) : Both normalized and non normalized are implemented.
- **Kurtosis** [[5]](#5)
- **Flatline**
- **Interlead Correlation coefficient** [[6]](#6)
- **Intralead Correlation coefficient** [[6]](#6)
- **Heart Rate**

More information about these indexes can be found in the article.

### Models :

The following models are implemented :
- Logistic regression (using **Sklearn** [[7]](#7))
- Custom Logistic regression
- LightGBM Tree (using **Light GBM library** [[8]](#8))

## Run the code :

### Activate Python environement :

After installing pdm and your environment, you must activate the python environement with the following commande line :

```python

    source .venv/bin/activate
```

### Get metrics data :

To get your metrics data (i.e. matrix of ECG_recordings*n_features), please run the following code line in your terminal :

```python

    python /workspaces/ecg_evaluation/src/operations/__exec_compute_metrics.py
```

in "__exec_compute_metrics.py", you must specify the path towards your data. Your data must be in Parquet format using [Petastorm](https://petastorm.readthedocs.io/en/latest/index.html).

The default path is to a gitignored data folder. You can insert your dataset in this folder and change the path to where the datafiles are.

The results will be stored in the "results" folder (in a netcdf (.nc) file format)

### Get Index evaluation done :

To Evaluate your index (with Stratified 10-fold Cross validation), please run the following code line in your terminal :


```python

    python /workspaces/ecg_evaluation/src/operations/__exec_evaluate_index.py
```

**The .nc file format must be already present! If not, run the previous part**

This will automatically evaluate each of the indexes and store the performance results into separate CSV files (stored in a folder called "evaluation_metrics") and the associated index score in separate **Pickle**[[9]](#9) files (in another folder called "proba_methods"). Files will be located into a folder "results" which **must** be gitignore.


## Models evaluation :

To run models evaluation and performance metrics, please run the following terminal command line :
```python

    python /workspaces/ecg_evaluation/src/operations/__exec_train_models.py
```

In "__exec_train_models.py", you can modiy the list of features you want to test and build your model on. You can also modified the type of model you want to try. You can also, by giving a list of features, applied a feature selection by changing the feature selection argument in the function by one of the 3 following :
- backward_selection
- JMI_score
- L2_regularization
- HJMI

The trained model prediction on the dataset will be stored in separate pickle files (in another folder called "proba_methods"). The name of your model will be indicated on the file.

The trained model perfomance on the dataset will be stored into separate CSV files (stored in a folder called "evaluation_metrics"). The name of your model will be indicated on the file.


## Visualization :

To visualize all of your results, please run the following notebook : **aggregate_results.ipynb**

All performance results ,for both indexes and models, will be printed as a table.
ROC and PR curve can also be printed with this notebook


## References

<a id="1">[1]</a>
Merkel, D. (2014). Docker: lightweight linux containers for consistent development and deployment. Linux Journal, 2014(239), 2.

<a id="2">[2]</a>
L. Targino Lins, “A fast signal quality assessment algorithm applied to a contactless electrocardiogram system,” masters, École de technologie supérieure, 2021. Accessed: Sep. 05, 2022. [Online]. Available: https://espace.etsmtl.ca/id/eprint/2747/

<a id="3">[3]</a>
Kramer L, Menon C, Elgendi M. ECGAssess: A Python-Based Toolbox to Assess ECG Lead Signal Quality. Front Digit Health. 2022 May 6;4:847555. doi: 10.3389/fdgth.2022.847555. PMID: 35601886; PMCID: PMC9120362.

<a id="4">[4]</a>
Takumi Sase, Jonatán Peña Ramírez, Keiichi Kitajo, Kazuyuki Aihara, Yoshito Hirata,
Estimating the level of dynamical noise in time series by using fractal dimensions,
Physics Letters A,
Volume 380, Issues 11–12,
2016,
Pages 1151-1163,
ISSN 0375-9601,
https://doi.org/10.1016/j.physleta.2016.01.014.
(https://www.sciencedirect.com/science/article/pii/S0375960116000177)

<a id="5">[5]</a>
P. Virtanen et al., “SciPy 1.0: fundamental algorithms for scientific computing in Python,”
Nat. Methods, vol. 17, no. 3, Art. no. 3, Mar. 2020, doi: 10.1038/s41592-019-0686-2.

<a id="6">[6]</a>
C. Orphanidou, Signal Quality Assessment in Physiological Monitoring. Cham: Springer
International Publishing, 2018. doi: 10.1007/978-3-319-68415-4.

<a id="7">[7]</a>
Pedregosa, F., Varoquaux, Ga"el, Gramfort, A., Michel, V., Thirion, B., Grisel, O., … others. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(Oct), 2825–2830.

<a id="8">[8]</a>
Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., … Liu, T.-Y. (2017). Lightgbm: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, 3146–3154.

<a id="9">[9]</a>
Van Rossum, G. (2020). The Python Library Reference, release 3.8.2. Python Software Foundation.
