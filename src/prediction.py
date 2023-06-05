import pandas as pd
import numpy as np

import os, sys, gc, yaml, glob
import logging.config

DATA_DIR = os.getenv('DATA_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
# LOG_DIR = os.getenv('LOG_DIR')
CONFIG_DIR = os.getenv('CONFIG_DIR')


class CreatePrediction:

    def __init__(self, config_d, type_label) -> None:
        self.config_d = config_d        
        # self.pred_click_type = setting_name
        self.type_label = type_label

    def main(self):
        '''購入リストを予測
        '''

        # load df to predict
        # 予測対象のdfを読み込み
        # validation or test
        test_df = self._load_test_df()
        test_df.sort_values(["session", "ts"], inplace= True)

        # # predict for click
        # pred_df_clicks = test_df.groupby(["session"]).apply(
        #     lambda x: suggest_clicks(x)
        # )
        # clicks_pred_df = pd.DataFrame(pred_df_clicks.add_suffix("_clicks"), columns=["labels"]).reset_index()
        # del pred_df_clicks
        # _ = gc.collect()

        # # predict for carts
        # carts_pred_df = pd.DataFrame(pred_df_buys.add_suffix("_carts"), columns=["labels"]).reset_index()

        # # predict for orders
        # orders_pred_df = pd.DataFrame(pred_df_buys.add_suffix("_orders"), columns=["labels"]).reset_index()

        # del pred_df_clicks, pred_df_buys
        # _ = gc.collect()

        # pred_df = pd.concat([clicks_pred_df, orders_pred_df, carts_pred_df])
        # pred_df.columns = ["session_type", "labels"]
        # pred_df["labels"] = pred_df.labels.apply(lambda x: " ".join(map(str,x)))

        output_path = os.path.join(OUTPUT_DIR, 'result', self.config_d['subfolder'], 'prediction.csv')
        # pred_df.to_csv(output_path, index=False)

        # del clicks_pred_df, orders_pred_df, carts_pred_df
        # _ = gc.collect()        

    def _load_test_df(self):
        '''data collection
        '''

        logging.info(f'start loading test_df')
        # data path取得
        if self.config_d['target'] == 'validation':
            files = sorted(glob.glob(
                os.path.join(DATA_DIR, 'validation', 'test_parquet', '*')
            ))
        elif self.config_d['target'] == 'test':
            files = sorted(glob.glob(
                os.path.join(DATA_DIR, 'chuncked_data', 'test_parquet', '*')
            ))
            
        # data読み込み
        adds = []
        for p in files:
            add = self._read_file_to_cache(p, self.type_label)
            adds.append(add)

        test_df = pd.concat(adds)
        logging.info(f'end loading data for test_df')
        return test_df
    
    def _read_file_to_cache(self, f, TYPE_LABEL):
        df = pd.read_parquet(f)
        df.ts = (df.ts/1000).astype('int32')
        df['type'] = df['type'].map(TYPE_LABEL).astype('int8')
        return df


# def suggest_clicks(df):
#     # USE USER HISTORY AIDS AND TYPES
#     aids=df.aid.tolist()
#     types = df.type.tolist()
#     unique_aids = list(dict.fromkeys(aids[::-1] ))
#     # RERANK CANDIDATES USING WEIGHTS
#     if len(unique_aids)>=20:
#         weights=np.logspace(0.1,1,len(aids),base=2, endpoint=True)-1
#         aids_temp = Counter() 
#         # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
#         for aid,w,t in zip(aids,weights,types): 
#             aids_temp[aid] += w * type_weight_multipliers[t]
#         sorted_aids = [k for k,v in aids_temp.most_common(20)]
#         return sorted_aids
#     # USE "CLICKS" CO-VISITATION MATRIX
#     aids2 = list(itertools.chain(*[top_20_time_w[aid] for aid in unique_aids if aid in top_20_time_w]))
#     # RERANK CANDIDATES
#     top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(20) if aid2 not in unique_aids]    
#     result = unique_aids + top_aids2[:20 - len(unique_aids)]
#     # USE TOP20 TEST CLICKS
#     return result + list(top_clicks)[:20-len(result)]

