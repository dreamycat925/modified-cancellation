# modified-cancellation

## Overview

This repository contains the statistical analysis code used in the validation study of the modified cancellation task, a novel neuropsychological assessment designed to evaluate attentional function using three stimulus types: geometric figures (Geo), numerical digits (Num), and technical symbols (Sym). The study employed Bayesian methods to assess the reliability, construct validity, and diagnostic utility of the modified cancellation task in differentiating patients with attentional impairments from healthy controls.

## Repository Contents  

The repository includes the following scripts, organized by task type and model type:


- **`Geo-Bayes-Logit.py`**: Bayesian logistic regression for geometric task classification
- **`GGeo-Bayes-Lin-Time.py`**: Bayesian linear regression for covariate adjustment (completion time)
- **`Geo-Bayes-Lin-Accuracy.py`**: Bayesian linear regression for covariate adjustment (accuracy)
- **`Num-Bayes-Logit.py`**: Bayesian logistic regression for numeric task classification
- **`Num-Bayes-Lin-Time.py`**: Bayesian linear regression for numeric task (completion time)
- **`Num-Bayes-Lin-Accuracy.py`**: Bayesian linear regression for numeric task (accuracy)
- **`Sym-Bayes-Logit.py`**: Bayesian logistic regression for symbol task classification
- **`Sym-Bayes-Lin-Time.py`**: Bayesian linear regression for symbol task (completion time)
- **`Sym-Bayes-Lin-Accuracy.py`**: Bayesian linear regression for symbol task (accuracy)

## Reproducibility

To ensure full reproducibility:

All dependencies and version information are listed in `requirements.txt`.

Analytical details (e.g., priors, model diagnostics, HPD ellipse derivation) are documented in the published manuscript and supplementary materials.

All models were implemented using PyMC.

## Citation

If you use this code or model in your research, please cite our paper:

```text
[Our Paper Title]
[Our Authors]
[Journal Name, Year]
[DOI or URL]
```
*Citation details will be added after the publication of our paper.*


## Note

- The repository does not include raw patient data due to privacy concerns.
- The statistical models were implemented using Bayesian inference with PyMC.
- The scripts assume that data preprocessing (e.g., reading Excel files, selecting relevant cases) has been completed beforehand.
- If you use this code, please cite the corresponding paper.

## License

This code is released under the MIT License.
