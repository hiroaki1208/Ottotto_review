import pandas as pd
import numpy as np

import os, sys
import argparse
import logging.config

def main():

    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__))
    )
    print('main')

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', default='validation'
                        , help= 'execution type(validation or test)'
                        , choices= ['validation', 'test'])
    args = parser.parse_args()

    try:
        logging.config.fileConfig(os.path.join(base_dir, 'logs', 'logging.ini'))
        logging.info(f'start: {base_dir}')
    except Exception as e:
        logging.exception(e)
        sys.exit(1)


    

if __name__ == '__main__':
    print('Hello')
    main()

