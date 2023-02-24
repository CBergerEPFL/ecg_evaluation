# ecg_evaluation
Project to evaluate ECG quality and noise level

## Project Goal :

This repo contains all the indexes and models implemented for ECG signal quality assessment. It was done in the context of this study (cite my master thesis?????).
The project was done in python using **Docker**[[1]](#1) with the docker image [PDM](https://hub.docker.com/r/frostming/pdm)

## Implementations :

### Index :

The indexes implemented are the following :
    1. **wPMF**[[2]](#2)
    2. **SNR**[[3]](#3) : This is an adaptation . We add a normalized version of this
    3. **TSD**[[4]](#4) : Both normalized and non normalized are implemented.
    3. **Kurtosis**[[5]](#5)
    4. Flatline
    5. **Interlead Correlation coefficient**[[6]](#6)
    6. **Intralead Correlation coefficient**[[6]](#6)
    7. Heart Rate

More information about these indexes can be found in the article.

### Models :

The following models are implemented :
    1. Logistic regression (using **Sklearn**[[7]](#7))
    2. Custom Logistic regression
    3. LightGBMTree (using **Light GBM library**[[8]](#8)

## Run the code :



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
