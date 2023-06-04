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
        # self.target = target
        # self.is_partial = is_partial

    def check_existing_file(self):
        '''既に既存のco-visitationファイルがあるかチェック
        '''

        

        return

    def data_collection(self, target, is_partial):
        '''data collection
        '''

        TYPE_LABEL = {'clicks':0, 'carts':1, 'orders':2}

        # data path取得
        if target == 'validation':
            self.files = sorted(glob.glob(
                os.path.join(DATA_DIR, 'validation', '*_parquet', '*')
            ))
        elif target == 'test':
            self.files = sorted(glob.glob(
                os.path.join(DATA_DIR, 'chuncked_data', '*_parquet', '*')
            ))

        # 一部取得の場合
        if bool(is_partial):
            self.files = random.choices(self.files, k= 12)
            
        # data読み込み
        self.data = {}
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

        TYPE_WEIGHT = {0:1, 1:6, 2:3}
        TOP = 50

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

            # 全ファイルを読み込んでいるかを確認
            assert set(self.files) == set(read_files)
            # convert matrix to dictionary
            tmp = tmp.reset_index()
            tmp = tmp.sort_values(['aid_x', 'wgt_y'], ascending= [True, False])
            # save top N
            tmp = tmp.reset_index(drop= True)
            tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
            tmp = tmp[tmp.n < TOP].copy()
            tmp.to_parquet()



            logging.info(f'end create co-visitaion chunck: {chunk}')







                    








        


        logging.info(f'end: create candidate by co-visitation')
            
