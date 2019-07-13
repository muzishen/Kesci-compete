
import pandas as pd
import numpy as np
test = pd.read_csv('./test_set.csv')
XGB1_result= pd.read_csv('./merge/submit_0.9406760360381264.csv')
XGB2_result= pd.read_csv('./merge/submit_0.9409034209801672.csv')
XGB3_result= pd.read_csv('./merge/submit_0.9409452361223127.csv')
# LGB1_result= pd.read_csv('./merge/LGB1.csv')
# LGB2_result= pd.read_csv('./merge/LGB2.csv')
# LGB3_result= pd.read_csv('./merge/LGB3.csv')
# LGB4_result= pd.read_csv('./merge/LGB4.csv')
# Stacking_result = pd.read_csv('result.csv')
XGB1_r =XGB1_result['pred']
XGB2_r =XGB1_result['pred']
XGB3_r =XGB1_result['pred']
# LGB1_r =LGB1_result['pred']
# LGB2_r =LGB2_result['pred']
# LGB3_r =LGB3_result['pred']
# LGB4_r =LGB4_result['pred']
# result_r =Stacking_result['pred']

result = pd.DataFrame()
test_id = test['ID'].values
result['ID'] = test_id
result['pred'] = (XGB1_r+XGB2_r+XGB3_r)/3
result.to_csv('merge.csv',index=False,sep=",")