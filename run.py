import pandas as pd
import numpy as np

import os, sys, yaml
import argparse
import logging.config

import models.create_covisit
import src.prediction

DATA_DIR = os.getenv('DATA_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
LOG_DIR = os.getenv('LOG_DIR')

def create_result_config(result_dir, args):
    '''計算設定を保存
    '''

    result_path = os.path.join(result_dir, 'result_config.yaml')
    with open(result_path, 'w') as rp:
        yaml.dump(vars(args), rp, default_flow_style=False)


def main():

    TYPE_LABEL = {'clicks':0, 'carts':1, 'orders':2}
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__))
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subfolder')
    parser.add_argument('-t', '--target', default='validation'
                        , help= 'execution type(validation or test)'
                        , choices= ['validation', 'test'])
    # parser.add_argument('-p', '--is_partial', default=1
    #                     , help= 'train data partial or not', type=int
    #                     )
    parser.add_argument('-c', '--covisit', default=1
                        , help= 'config of co-visitation matrix'
                        )
    args = parser.parse_args()

    try:
        logging.basicConfig(level=logging.INFO)   
        logging.config.fileConfig(os.path.join(base_dir, 'logs', 'logging.ini')
                                  , defaults={'logdir': LOG_DIR})
        
        # 計算設定保存
        result_config_dir = os.path.join(OUTPUT_DIR, 'result', args.subfolder)
        os.mkdir(result_config_dir)
        create_result_config(result_config_dir, args)

        logging.info(f'start: {base_dir}')
        logging.info(f'(param)target: {args.target}')
        # logging.info(f'(param)is_partial: {args.is_partial}')
        logging.info(f'(param)covisitation setting: {args.covisit}')

        # 特徴量

        # create model
        ## co-visitaion
        CoVisitaion = models.create_covisit.CreateCoVisitaion(
            args.covisit, TYPE_LABEL
            )
        CoVisitaion.main()

        # prediction
        Prediction = src.prediction.CreatePrediction(vars(args), TYPE_LABEL)
        Prediction.main()



        logging.info(f'end: {base_dir}')
    except Exception as e:
        logging.exception(e)
        sys.exit(1)


if __name__ == '__main__':
    # print('Hello')
    main()

