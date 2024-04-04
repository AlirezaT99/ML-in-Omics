from ..analysis.preprocessing import *
from ..analysis.models import *

DEFAULT_DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'
OUTPUT_LOG_FILE = 'log.txt'
DEFAULT_DEBUG_MODE = False

NA_THRESHOLD = 0.90
RANDOM_STATE = 42
KFOLD_SPLITS = 5
DEFAULT_SCORING = 'f1_weighted'

ANALYSIS_PARAMS = {
    'mars_ibs_lr': {
        'path': 'MARS_IBS_2020',
        'preprocessor': PC,
        'preproc_kwargs': {'var': 0.9},
        'classifier': LogisticRegression,
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
}