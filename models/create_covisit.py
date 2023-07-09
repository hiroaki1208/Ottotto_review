import pandas as pd
import numpy as np
import glob
import random

import os, sys, gc, yaml
import logging.config

DATA_DIR = os.getenv('DATA_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
# LOG_DIR = os.getenv('LOG_DIR')
CONFIG_DIR = os.getenv('CONFIG_DIR')
random.seed(2023)

class CreateCoVisitaion():

    def __init__(self, param, type_label) -> None:
        self.param = param
        self.type_label = type_label
        self.files = None
        self.output_dir_path = None
        self.is_cudf= False

    def _data_collection(self):
        '''data collection
        '''

        print(f'start loading data for create co-visitaion')
        # data path取得
        if self.param['target'] == 'validation':
            if self.is_cudf:
                self.files = sorted(glob.glob('../input/otto-validation/*_parquet/*'))
            else:
                self.files = sorted(glob.glob(
                    os.path.join(DATA_DIR, 'validation', '*_parquet', '*')
                ))

        elif self.param['target'] == 'test':
            if self.is_cudf:
                self.files = sorted(glob.glob('../input/otto-chunk-data-inparquet-format/train_parquet/*'))
            else:
                self.files = sorted(glob.glob(
                    os.path.join(DATA_DIR, 'chuncked_data', '*_parquet', '*')
                ))

        # 一部取得の場合
        self.files = random.choices(self.files, k= 7)
            
        # data読み込み
        self.data = {}
        if self.is_cudf:
            for f in self.files: self.data[f] = cudf.DataFrame( self._read_file_to_cache(f, self.type_label) )
        else:
            for p in self.files: self.data[p] = self._read_file_to_cache(p, self.type_label)
        print(f'end loading data for create co-visitaion')

    def _read_file_to_cache(self, f, TYPE_LABEL):
        df = pd.read_parquet(f)
        df.ts = (df.ts/1000).astype('int32')
        df['type'] = df['type'].map(TYPE_LABEL).astype('int8')
        return df
    
    def _covisitation_type_weight(self, df, part, SIZE, target):
        '''type_weightでのco-visitation matirix作成詳細
        '''

        # session x aid ごとに行動回数を重みづけしてsum
        df['wgt'] = df['type'].map(self.param['covisitation'][target]['type_weight'])
        df['wgt'] = df['wgt'].astype('float32')
        df = df.groupby(['session', 'aid']).wgt.sum()
        df = df.reset_index()

        # 同時に購入されたpair
        ## メモリのためにaidを一部に絞る
        ## その分partをPART回数ループ
        df_merge = df.loc[(df.aid >= part*SIZE) & (df.aid < (part+1)*SIZE)].copy()
        ## マージする先はaidを一部に絞らない(絞った対象以外のペアが取れなくなる)
        df = df_merge.merge(df, on= 'session')
        del df_merge
        _= gc.collect()

        df = df.loc[ (df.aid_x != df.aid_y)].copy() # 同じaidのペアを削除
        df = df[['aid_x', 'aid_y', 'wgt_y']].copy() # ペアとなる先のwgtが重要
        df = df.groupby(['aid_x', 'aid_y']).wgt_y.sum() # ペアごとに集計

        return df
    
    def _covisitation_b2b(self, df, part, SIZE):
        '''buy2buyでのco-visitation matirix作成詳細
        '''

        # session x aid ごとに行動回数を重みづけしてsum
        # cart or buyのみに限定
        df = df[df['type'].isin([1,2])].copy()
        
        df['wgt'] = df['type'].map({1:1, 2:1})
        df['wgt'] = df['wgt'].astype('float32')
        df = df.groupby(['session', 'aid']).wgt.sum()
        df = df.reset_index()

        # 同時に購入されたpair
        ## メモリのためにaidを一部に絞る
        ## その分partをPART回数ループ
        df_merge = df.loc[(df.aid >= part*SIZE) & (df.aid < (part+1)*SIZE)].copy()
        ## マージする先はaidを一部に絞らない(絞った対象以外のペアが取れなくなる)
        df = df_merge.merge(df, on= 'session')
        del df_merge
        _= gc.collect()

        df = df.loc[ (df.aid_x != df.aid_y)].copy() # 同じaidのペアを削除
        df = df[['aid_x', 'aid_y', 'wgt_y']].copy() # ペアとなる先のwgtが重要
        df = df.groupby(['aid_x', 'aid_y']).wgt_y.sum() # ペアごとに集計

        return df

    def _covisitation_time_weight(self, df, part, SIZE):
        '''time weightでのco-visitation matirix作成詳細
        '''

        # 同時に購入されたpairを先に作成
        ## メモリのためにaidを一部に絞る
        ## その分partをPART回数ループ
        df_merge = df.loc[(df.aid >= part*SIZE) & (df.aid < (part+1)*SIZE)].copy()
        ## マージする先はaidを一部に絞らない(絞った対象以外のペアが取れなくなる)
        df = df_merge.merge(df, on= 'session')
        del df_merge
        _= gc.collect()

        df = df.loc[ (df.aid_x != df.aid_y)].copy() # 同じaidのペアを削除

        # ts_xで重みづけ
        # TODO: 重みづけのウェイト修正
        if self.param['target'] == 'validation':
            min_ts = 1659304800
            max_ts = 1662328791
        elif self.param['target'] == 'configuration':
            min_ts = 1659304800
            max_ts = 1662328791
        df['wgt_y'] = 1 + 3*(df.ts_x - min_ts)/(max_ts - min_ts)
        df = df[['aid_x', 'aid_y', 'wgt_y']].copy() # ペアとなる先のwgtが重要
        df = df.groupby(['aid_x', 'aid_y']).wgt_y.sum() # ペアごとに集計

        return df

    def _create_covisit_candidate(self, target):
        '''Co-Visitaion履歴から候補作成
        chunkなどでの読み込みまではnameごとに同じ
        読み込んだ後の処理をnameごとに変更
            - type_weight: typeごとに重みづけして算出
        '''

        DISK_SIZE = 4
        SIZE = 1.86e6 / DISK_SIZE

        CHUNK_SIZE = 6
        READ_CT = 5
        CHUNK_LEN = int( np.ceil( len(self.files)/CHUNK_SIZE ) )

        # co-visitation matrixの種類(type weight, b2b, time_weight)
        name = self.param['covisitation'][target]['name']

        print(' ================== ')
        print(f'start: create candidate by co-visitation for {target} name: {name}')

        # メモリの関係上4つに分ける
        # ペア作成元のaidを4分割
        for part in range(DISK_SIZE):
            # if part > 0: break
            print(f'start create co-visitaion part: {part}')

            # 全sessionではなく一部sessionに絞る(メモリ対策)
            # ファイルの読み込み数を限定
            # 限定１：a,bの範囲ないに限定(このまとまりをchunck)            
            read_files = []
            for chunk in range(CHUNK_SIZE):
                print(f'start create co-visitaion chunck: {chunk} in part {part}')

                a = chunk * CHUNK_LEN
                b = min( (chunk+1)*CHUNK_LEN, len(self.files))

                # b > aの場合早期リターン
                if b < a:
                    continue

                # 限定２：a,bの範囲内で、更にREAD_CTごとに限定
                # kは起点、kからREAD_CT分のファイルを取得
                for k in range(a,b,READ_CT):
                    print(f'start create co-visitaion start from {k} in chunck {chunk}')

                    adds = []
                    for i in range(READ_CT):
                        # indexがbを超えたらbreak
                        # 今回読み込む範囲は限定２のa,bなので
                        if k+i >= b: break

                        path = self.files[k+i]
                        add = self.data[path]
                        adds.append(add)
                        read_files.append(path)

                    if self.is_cudf:
                        df = cudf.concat(adds)
                    else:
                        df = pd.concat(adds)

                    # setting内のnameごとに処理を分ける
                    if name == 'type_weight':
                        df = self._covisitation_type_weight(df, part, SIZE, target)
                    if name == 'buy2buy':
                        df = self._covisitation_b2b(df, part, SIZE)
                    if name == 'time_weight':
                        df = self._covisitation_time_weight(df, part, SIZE)
                    
                    # 限定２の中で合計
                    if k == a: tmp2 = df
                    else: tmp2 = tmp2.add(df, fill_value= 0)
                    print(f'end create co-visitaion start from {k} in chunck {chunk}')
                
                # 限定１の中で合計
                if chunk == 0: tmp = tmp2
                else: tmp = tmp.add(tmp2, fill_value= 0)
                del tmp2, df
                _ = gc.collect()
                print(f'end create co-visitaion chunck: {chunk} in part {part}')

            # 全ファイルを読み込んでいるかを確認
            assert set(self.files) == set(read_files)
            # convert matrix to dictionary
            tmp = tmp.reset_index()
            tmp = tmp.sort_values(['aid_x', 'wgt_y'], ascending= [True, False])
            # save top N
            tmp = tmp.reset_index(drop= True)
            tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
            tmp = tmp[tmp.n < self.param['covisitation'][target]['N']].copy()

            # save file
            # partごとに保存
            if self.is_cudf:
                output_path = f'{target}_part{part}.pqt'
                tmp.to_pandas().to_parquet(output_path)
            else:
                output_path = os.path.join(self.output_dir_path, f'part{part}.pqt')
                tmp.to_parquet(output_path)
            del tmp
            _ = gc.collect()

            print(f'end create co-visitaion part: {part}')

        print(f'end: create candidate by co-visitation for {target} name: {name}')
        print(' ================== ')

    def main(self):
        '''
        既に既存のco-visitationファイルがあるかチェック
        -> ない場合、co-visitaion作成を実行
        '''
        
        # prediction対象でfor
        prediction_targets = list( self.param['prediction'].keys() )
        for prediction_target in prediction_targets:

            # covisitaion matrixの設定でfor
            covisitation_targets = self.param['prediction'][prediction_target]['covisitation']['target']
            for covisitation_target in covisitation_targets:

                # model出力先ディレクトリ
                if not self.is_cudf:
                    self.output_dir_path = os.path.join(
                        OUTPUT_DIR, self.param['target'], 'model', 'covisit', f'{covisitation_target}'
                        )
                    if os.path.exists(self.output_dir_path):
                        print(f'co-visit {covisitation_target} is already exits(now {prediction_target})')
                        continue
                    else:
                        os.mkdir(self.output_dir_path)

                print(f'start create co-visit {covisitation_target}')
                if not self.is_cudf:
                    os.mkdir(self.output_dir_path)

                # data collectionの対象はself.targetによる
                # class内で一度でも収集していればよい
                if self.files is None:
                    self._data_collection()
                self._create_covisit_candidate(covisitation_target)

                print(f'end create co-visit {covisitation_target}')

