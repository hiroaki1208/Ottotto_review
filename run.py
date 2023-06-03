import pandas as pd
import numpy as np

import os, sys
import argparse
import logging.config

DATA_DIR = os.getenv('DATA_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
LOG_DIR = os.getenv('LOG_DIR')

def main():

    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__))
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', default='validation'
                        , help= 'execution type(validation or test)'
                        , choices= ['validation', 'test'])
    args = parser.parse_args()

    try:
        logging.basicConfig(level=logging.INFO)   
        logging.config.fileConfig(os.path.join(base_dir, 'logs', 'logging.ini')
                                  , defaults={'logdir': LOG_DIR})
        
        logging.info(f'start: {base_dir}')
        logging.info(f'param_target: {args.target}')

        logging.info(f'end: {base_dir}')
    except Exception as e:
        logging.exception(e)
        sys.exit(1)


    

if __name__ == '__main__':
    # print('Hello')
    main()

