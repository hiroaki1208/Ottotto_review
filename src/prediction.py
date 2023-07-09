import pandas as pd
import numpy as np

import os, gc, glob, sys
from collections import Counter
import itertools, datetime
from copy import deepcopy

sys.path.append(os.getenv('SRC_DIR'))
import prediction_ensamble_func

DATA_DIR = os.getenv('DATA_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
# LOG_DIR = os.getenv('LOG_DIR')
CONFIG_DIR = os.getenv('CONFIG_DIR')


class CreatePrediction:

    def __init__(self, param, type_label, param_idx) -> None:
        self.param = param
        self.type_label = type_label
        self.param_idx = param_idx

        self.ensemble_func = {
            'ensemble0': self._prediction_covisit_ensamble0,
            'ensemble1': self._prediction_covisit_ensamble1,
        }

        self.test_df = pd.DataFrame()
        self.covisit_dict = {}

    def _add_now_time(self, msg):
        dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        return f'{dt}: {msg}'
    
    def _load_test_df(self):
        '''data collection
        '''
        print(f'start loading df to predict')

        # data path取得
        if self.param['target'] == 'validation':
            files = sorted(glob.glob(
                os.path.join(DATA_DIR, 'validation', 'test_parquet', '*')
            ))
        elif self.param['target'] == 'test':
            files = sorted(glob.glob(
                os.path.join(DATA_DIR, 'chuncked_data', 'test_parquet', '*')
            ))
            
        # data読み込み
        adds = []
        for p in files:
            add = self._read_file_to_cache(p, self.type_label)
            adds.append(add)

        self.test_df = pd.concat(adds)
        self.test_df.sort_values(["session", "ts"], inplace= True)
        print(f'end loading df to predict')
    
    def _read_file_to_cache(self, f, TYPE_LABEL):
        df = pd.read_parquet(f)
        df.ts = (df.ts/1000).astype('int32')
        df['type'] = df['type'].map(TYPE_LABEL).astype('int8')
        return df

    def _collect_covisit_setting(self, target):
        '''予測用co-visitation matrix取得
        co-visitation matrixを取得
        辞書に変換
        TODO: covisit topNのみ抽出をパラメータ投入？
        TODO: 辞書型データを保存？
        '''
        print('start: collect co-visitation matrix as dict')

        # target co-visitationを取得
        targets = self.param['prediction'][target]['covisitation']['target']
        
        for target in targets:
            print(f'start: collect co-visitation matrix as dict target: {target}')

            if target in self.covisit_dict.keys():
                print(f'collect co-visitation matrix {target} is already as dict')
                continue
            
            # co-visitation matrix path取得
            input_dir = os.path.join(
                OUTPUT_DIR, self.param['target'], 'model', 'covisit', target
                )
            input_paths = sorted( glob.glob(
                os.path.join(input_dir, '*')
                  ))

            # 辞書型で読み込み
            covisit_matrix_as_dict = {}
            for i, path in enumerate(input_paths):
                _tmp = pd.read_parquet(path)
                _tmp = _tmp.groupby('aid_x').aid_y.apply(list).to_dict()
                if i == 0:
                    covisit_matrix_as_dict = _tmp
                else:
                    covisit_matrix_as_dict.update(_tmp)
                del _tmp
                _ = gc.collect()
            
            # 結果を保存
            self.covisit_dict[target] = covisit_matrix_as_dict
            del covisit_matrix_as_dict
            _ = gc.collect()
            print(f'end: collect co-visitation matrix as dict target: {target}')

        print('end: collect co-visitation matrix as dict')

    def _prediction_covisit_ensamble0(self, target):
        '''prediction by ensamble0
        '''
        print(self._add_now_time(f'start prediction by ensamble0 for {target}'))        

        # prediction
        pred = self.test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
            # lambda x: self._suggest_clicks(x)
            lambda x: prediction_ensamble_func._suggest_clicks(x, self.covisit_dict['setting0'])
        )

        pred = pd.DataFrame(pred.add_suffix(f"_{target}"), columns=["labels"]).reset_index()
        path = os.path.join(
            os.path.join(self.output_dir, f'prediction_{target}.parquet')
        )
        pred.to_parquet(path)
        print(self._add_now_time(f'end prediction by ensamble0 for {target}'))        
        return pred

    def _prediction_covisit_ensamble1(self, target):
        '''prediction by ensamble1
        過去購入aidをそのまま返す
        '''
        print(self._add_now_time(f'start prediction by ensamble1 for {target}'))

        # prediction
        pred = self.test_df.sort_values(["session", "ts"]).groupby(["session"]).apply(
            lambda x: prediction_ensamble_func._past_aid_only(x)
        )

        pred = pd.DataFrame(pred.add_suffix(f"_{target}"), columns=["labels"]).reset_index()
        path = os.path.join(
            os.path.join(self.output_dir, f'prediction_{target}.parquet')
        )
        pred.to_parquet(path)
        print(self._add_now_time(f'end prediction by ensamble1 for {target}'))
        return pred

    def _validate(self):
        '''validation score
        '''

        # load dataset
        ## prediction
        path = os.path.join(
            os.path.join(self.output_dir, 'prediction.csv')
        )
        pred_df = pd.read_csv(path)
        ## test_ctrl label
        path = os.path.join(
            os.path.join(DATA_DIR, 'validation', 'test_labels.parquet')
        )
        test_labels_row = pd.read_parquet(path)
            
        # prediction対象でfor
        score_d = {}
        score = 0
        weights = {'clicks': 0.10, 'carts': 0.30, 'orders': 0.60}
        prediction_targets = list( self.param['prediction'].keys() )
        for prediction_target in prediction_targets:
            
            if prediction_target == 'clicks':
                continue

            sub = pred_df.loc[pred_df.session_type.str.contains(prediction_target)].copy()
            sub['session'] = sub.session_type.apply(lambda x: int(x.split('_')[0]))
            # filter top 20
            sub.labels = sub.labels.apply(lambda x: [int(i) for i in x.split(' ')[:20]])

            test_labels = deepcopy(test_labels_row)
            test_labels = test_labels.loc[test_labels['type']==prediction_target]
            test_labels = test_labels.merge(sub, how='left', on=['session'])

            test_labels['hits'] = test_labels.apply(lambda df: len(set(df.ground_truth).intersection(set(df.labels))), axis=1)
            test_labels['gt_count'] = test_labels.ground_truth.str.len().clip(0,20)
            test_labels['label_count'] = test_labels.labels.str.len()
            path = os.path.join(
                os.path.join(self.output_dir, f'score_detail_{prediction_target}.parquet')
            )
            test_labels.to_parquet(path)

            recall = test_labels['hits'].sum() / test_labels['gt_count'].sum()
            print(f'{prediction_target} score(recall in 20) =',recall)
            score_d[prediction_target] = recall
            score += weights[prediction_target]*recall

        score_d['total_score'] = score
        score_df = pd.DataFrame(score_d.values(), index= score_d.keys(), columns= ['score'])
        path = os.path.join(
            os.path.join(self.output_dir, f'score_all.csv')
        )
        score_df.to_parquet(path)

    def main(self):
        '''購入リストを予測
        予測用dfを読み込み
            - test or validation df
        予測用model読み込み
            - co-visitation matrix
            - dict形式に変換
        予測
            - co visitation matrixからtop20抽出方法
            - ログ
        評価(validationのみ)
            

        #TODO
        - 既に対象の設定の予測結果があるかを確認
            - ない場合に以下を実施
        '''

        # prediction保存先
        self.output_dir = os.path.join(
            os.path.join(OUTPUT_DIR, self.param['target'], 'result', self.param_idx)
        )
        if os.path.exists(self.output_dir):
            print('result is already exists')
            return
        else:
            os.mkdir(self.output_dir)

        # 予測用dfを読み込み
        self._load_test_df()

        # prediction対象でfor
        adds = []
        prediction_targets = list( self.param['prediction'].keys() )
        for prediction_target in prediction_targets:
            # TODO: covisitationで予測するのが前提

            # 使用co-visitation matrixを読み込み
            self._collect_covisit_setting(prediction_target)

            # prediction            
            add = self.ensemble_func[
                self.param['prediction'][prediction_target]['covisitation']['ensemble']
            ](prediction_target)

            adds.append(add)
            del add
            _ = gc.collect()

        pred_df = pd.concat(adds)
        del adds
        _ = gc.collect()

        pred_df.columns = ["session_type", "labels"]
        pred_df["labels"] = pred_df.labels.apply(lambda x: " ".join(map(str,x)))
        path = os.path.join(
            os.path.join(self.output_dir, 'prediction.csv')
        )
        pred_df.to_csv(path, index=False)
        del pred_df
        _ = gc.collect()

        # validation
        if self.param['target'] == 'validation':
            self._validate()
        
