import pandas as pd
import numpy as np
import glob
import random

import os, sys, gc
import logging.config

DATA_DIR = os.getenv('DATA_DIR')
# OUTPUT_DIR = os.getenv('OUTPUT_DIR')
# LOG_DIR = os.getenv('LOG_DIR')


class CreateCoVisitaion():

    # def __init__(self) -> None:
    # def __init__(self, target, is_partial) -> None:
        # self.target = target
        # self.is_partial = is_partial


    # def data_collection(self):
    def data_collection(self, target, is_partial):
        '''data collection
        '''

        TYPE_LABEL = {'clicks':0, 'carts':1, 'orders':2}

        # data path取得
        if target == 'validation':
            # self.files_train = sorted(glob.glob(
            #     os.path.join(DATA_DIR, 'validation', 'train_parquet', '*')
            # ))
            # self.files_test = sorted(glob.glob(
            #     os.path.join(DATA_DIR, 'validation', 'test_parquet', '*')
            # ))
            self.files = sorted(glob.glob(
                os.path.join(DATA_DIR, 'validation', '*_parquet', '*')
            ))
        elif target == 'test':
            self.files_train = sorted(glob.glob(
                os.path.join(DATA_DIR, 'chunked_date', 'train_parquet', '*')
            ))
            self.files_test = sorted(glob.glob(
                os.path.join(DATA_DIR, 'chunked_data', 'test_parquet', '*')
            ))

        # 一部取得の場合
        if bool(is_partial):
            # self.files_train = self.files_train[:10]
            # self.files_test = self.files_test[:10]
            self.files = random.choices(self.files, k= 12)
            
        # data読み込み
        # self.data_train = {}
        # self.data_test = {}
        self.data = {}
        # for p in self.files_train: self.data_train[p] = self._read_file_to_cache(p, TYPE_LABEL)
        # for p in self.files_test: self.data_test[p] = self._read_file_to_cache(p, TYPE_LABEL)
        for p in self.files: self.data[p] = self._read_file_to_cache(p, TYPE_LABEL)

    def _read_file_to_cache(self, f, TYPE_LABEL):
        df = pd.read_parquet(f)
        df.ts = (df.ts/1000).astype('int32')
        df['type'] = df['type'].map(TYPE_LABEL).astype('int8')
        return df

    def covisit_candidate(self):
        '''Co-Visitaion履歴から候補作成
        '''


        DISK_SIZE = 4
        SIZE = 1.86e6 / DISK_SIZE

        CHUNK_SIZE = 6
        READ_CT = 5
        CHUNK_LEN = int( np.ceil( len(self.files)/CHUNK_SIZE ) )
        print(f'chunk length: {CHUNK_LEN}')

        TYPE_WEIGHT = {0:1, 1:6, 2:3}

        logging.info(f'start: create candidate by co-visitation')

        # メモリの関係上4つに分ける
        # ペア作成元のaidを4分割
        for part in range(DISK_SIZE):
            if part > 0: break
            logging.info(f'start create co-visitaion part: {part}')

            # 全sessionではなく一部sessionに絞る(メモリ対策)
            # ファイルの読み込み数を限定
            # 限定１：a,bの範囲ないに限定(このまとまりをchunck)            
            read_files = []
            for chunk in range(CHUNK_SIZE):
                logging.info(f'start create co-visitaion chunck: {chunk}')

                a = chunk * CHUNK_LEN
                b = min( (chunk+1)*CHUNK_LEN, len(self.files))

                # 限定２：a,bの範囲内で、更にREAD_CTごとに限定
                # kは起点、kからREAD_CT分のファイルを取得
                for k in range(a,b,READ_CT):
                    logging.info(f'start create co-visitaion start from {k} in chunck {chunk}')

                    adds = []
                    for i in range(READ_CT):
                        # indexがbを超えたらbreak
                        # 今回読み込む範囲は限定２のa,bなので
                        if k+i >= b: break

                        path = self.files[k+i]
                        add = self.data[path]
                        adds.append(add)
                        read_files.append(path) 
                    df = pd.concat(adds)

                    # session x aid ごとに行動回数を重みづけしてsum
                    df['wgt'] = df['type'].map(TYPE_WEIGHT)
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

                    # 限定２の中で合計
                    if k == a: tmp2 = df
                    else: tmp2 = tmp2.add(df, fill_value= 0)
                    logging.info(f'end create co-visitaion start from {k} in chunck {chunk}')
                
                # 限定１の中で合計
                if chunk == 0: tmp = tmp2
                else: tmp = tmp.add(tmp2, fill_value= 0)
                del tmp2, df
                _ = gc.collect()

            assert set(self.files) == set(read_files)
            logging.info(f'end create co-visitaion chunck: {chunk}')







                    








        


        logging.info(f'end: create candidate by co-visitation')
            
