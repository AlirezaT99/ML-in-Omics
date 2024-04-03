import os
from argparse import ArgumentParser

import warnings
warnings.filterwarnings('ignore')

from analysis.analysis import Analysis
from utils.consts import DEFAULT_DATA_PATH, DEFAULT_DEBUG_MODE, ANALYSIS_PARAMS


def main():
    parser = ArgumentParser(description='ML in Omics analysis configuration')
    parser.add_argument('--path', '-p', default=DEFAULT_DATA_PATH, type=str, help='Path to data folder')
    parser.add_argument('--study', '-s', type=str, default=None, help='')
    parser.add_argument('--debug', '-d', type=bool, default=DEFAULT_DEBUG_MODE, help='Debug mode')
    
    args = parser.parse_args()

    if args.debug:
        print(args)
    
    if not args.study or args.study not in ANALYSIS_PARAMS:
        print(f'Please provide a valid study: {ANALYSIS_PARAMS.keys()}')
        return
    
    if not os.path.exists(args.path):
        print(f'Path {args.path} does not exist')
        return

    Analysis(args.path, args.study, args.debug, **ANALYSIS_PARAMS[args.study]).run()

if __name__ == '__main__':
    main()
