# ML in Omics
Project Topic: Machine Learning in Omics: Integration of Metagenomics and Metabolomics.

Source code and documentation for Life Science Technologies Project Course B, Aalto University, Finland.

## How to Run
### Environment Setup
(Optional) First, create a virtual environment:
```bash
python -m venv mlomics         # or `virtualenv mlomics`
source mlomics/bin/activate    # to activate the env
```

Next, install the requirements:
```bash
pip install -r requirements.txt
```

### Analysis setup
First, the configuration for the study must be available in the [configuration file](./utils/consts.py). The config for each study looks something like this:

```json
'mars_ibs_lr': {
    'path': 'MARS_IBS_2020',
    'preprocessor': FeatureExtraction.PC,
    'preproc_kwargs': {'var': 0.9},
    'classifier': Models.LogisticRegression,
    'classifier_param_grid': {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200, 300],
        'tol': [0.0001, 0.001, 0.01],
        'class_weight': [None, 'balanced'],
        'random_state': [42]
    },
},
```

Then, you can run the analysis using the following command and arguments:

```bash
python main.py --path "path/to/data" --study "mars_ibs_lr" --output "../output" --debug True
```

## Authors
Alexandra Gorbonos

Alireza Tajmirriahi

Mohammad Sherafati

Salma Rachidi
