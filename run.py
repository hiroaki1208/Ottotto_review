import pandas as pd
import numpy as np

import os, sys, yaml
import argparse
import logging.config

import models.create_covisit
import src.prediction

DATA_DIR = os.getenv('DATA_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
CONFIG_DIR = os.getenv('CONFIG_DIR')
LOG_DIR = os.getenv('LOG_DIR')

def create_result_config(result_dir, args):
    '''計算設定を保存
    '''

    result_path = os.path.join(result_dir, 'result_config.yaml')
    with open(result_path, 'w') as rp:
        yaml.dump(vars(args), rp, default_flow_style=False)

def _load_config_file(fname):
    '''parameter設定ファイルを読み込み
    '''
    logging.info(f'start loading parameter configuration')
    with open(os.path.join(CONFIG_DIR, 'parameter', f'{fname}.yml')) as file:
        config = yaml.safe_load(file.read())
    # config = config_covisit[self.setting_name]
    logging.info(f'end loading parameter configuration')

    return config

def main():

    TYPE_LABEL = {'clicks':0, 'carts':1, 'orders':2}
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__))
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subfolder')
    # parser.add_argument('-t', '--target', default='validation'
    #                     , help= 'execution type(validation or test)'
    #                     , choices= ['validation', 'test'])
    # parser.add_argument('-p', '--is_partial', default=1
    #                     , help= 'train data partial or not', type=int
    #                     )
    parser.add_argument('-p', '--param_idx', default='param0'
                        , help= 'parameter index', type=str
                        )
    # parser.add_argument('-c', '--covisit', default=1
    #                     , help= 'config of co-visitation matrix'
    #                     )
    args = parser.parse_args()

    # parameter設定yml読み込み
    param_config = _load_config_file(args.param_idx)
    # param_config['type_label'] = TYPE_LABEL
    import pdb; pdb.set_trace()

    try:
        logging.basicConfig(level=logging.INFO)   
        logging.config.fileConfig(os.path.join(base_dir, 'logs', 'logging.ini')
                                  , defaults={'logdir': LOG_DIR})
        
        # # 計算設定保存
        # result_config_dir = os.path.join(OUTPUT_DIR, 'result', args.subfolder)
        # os.mkdir(result_config_dir)
        # create_result_config(result_config_dir, args)

        logging.info(f'start: {base_dir}')
        # logging.info(f'(param)target: {args.target}')
        # logging.info(f'(param)is_partial: {args.is_partial}')
        # logging.info(f'(param)covisitation setting: {args.covisit}')

        # 特徴量

        # create model
        ## co-visitaion
        # CoVisitaion = models.create_covisit.CreateCoVisitaion(
        #     param_config['covisitation'], TYPE_LABEL
        #     )
        # CoVisitaion.main()

        # # prediction
        # Prediction = src.prediction.CreatePrediction(
        #     vars(args), TYPE_LABEL, param_config
        #     )
        # Prediction.main()



        logging.info(f'end: {base_dir}')
    except Exception as e:
        logging.exception(e)
        sys.exit(1)


if __name__ == '__main__':
    # print('Hello')
    main()

