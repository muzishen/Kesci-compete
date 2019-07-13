import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import warnings
np.random.seed(42)
warnings.filterwarnings('ignore')

train = pd.read_csv('./train_set.csv')
test = pd.read_csv('./test_set.csv')

data = pd.concat([train,test],axis=0)

# 简单的预处理操作
'''
月份映射
日期叠加
类别特征替换为数字
类别特征字段 unknow 转化为 others
'''
data['month'] = data['month'].map({
            'feb':2,'may':5,'apr':4,'jul':7,'jun':6,'nov':11,
            'aug':8,'jan':1,'dec':12,'oct':10,'sep':9,'mar':3
            })
day_map = {0:0,1:31,2:59,3:90,4:120,5:151,6:181,
           7:212,8:243,9:273,10:304,11:334,12:365}

long_feat1 = []
long_feat2 = []
long_feat3 = []

for i in data[['day','month','pdays']].values:
    today = i[0] + day_map[i[1]-1]
    # long_feat2.append(i[0]/365)
    long_feat1.append(today)
    long_feat3.append(today/365)

data['long_feat1'] = long_feat1
# data['long_feat2'] = long_feat2
data['long_feat3'] = long_feat3

data['month_day'] = data['month'] * 100 + data['day']
data['1/month'] = 1 / data['month']
data['1/day'] = 1 / data['day']

data['loan'] = data['loan'].map({'yes':1,'no':0})
data['housing'] = data['housing'].map({'yes':1,'no':0})
data['default'] = data['default'].map({'yes':1,'no':0})

data['poutcome'] = data['poutcome'].replace('unknown','other')
data['contact'] = data['contact'].replace('unknown','other')
data['education'] = data['education'].replace('unknown','other')
data['job'] = data['job'].replace('unknown','other')

feature=data.columns.tolist()
feature.remove('ID')
feature.remove('y')
sparse_feature= ['campaign','contact','default','education','housing','job','loan','marital','month','poutcome']
dense_feature= list(set(feature) - set(sparse_feature))

# 统计月份 具体天数 比例特征族
for f in['campaign', 'contact','default','education','housing','job','loan','marital','poutcome']:
    data['count_day_month_{}'.format(f)] = data.groupby(['day','month',f])[f].transform('count')
    # data['count_pdays_{}'.format(f)] = data.groupby(['pdays',f])[f].transform('count')
    data['count_month_{}'.format(f)] = data.groupby(['month',f])[f].transform('count')
    data['count_day_month_{}/count_month_{}'.format(f,f)] = data['count_month_{}'.format(f)] / data['count_day_month_{}'.format(f)]

def get_new_columns(name,aggs):
    l=[]
    for k in aggs.keys():
        for agg in aggs[k]:
            if str(type(agg))=="<class 'function'>":
                l.append(name + '_' + k + '_' + 'other')
            else:
                l.append(name + '_' + k + '_' + agg)
    return l

for d in sparse_feature:
    aggs={}
    for s in sparse_feature:
        aggs[s]=['count','nunique']
    for den in dense_feature:
        aggs[den]=['mean','max','min','std']
    aggs.pop(d)
    temp=data.groupby(d).agg(aggs).reset_index()
    temp.columns=[d]+get_new_columns(d,aggs)
    data=pd.merge(data,temp,on=d,how='left')

def cap(x,quantile=[0.01,0.99]):
    Q01,Q99=x.quantile(quantile).values.tolist()
    if Q01 > x.min():
        x = x.copy()
        x.loc[x<Q01] = Q01
    if Q99 < x.max():
        x = x.copy()
        x.loc[x>Q99] = Q99
    return(x)

for s in ['campaign','contact','default','education','housing','job','loan','marital','month','poutcome']:
    data=pd.concat([data,pd.get_dummies(data[s],prefix=s+'_')],axis=1)
    data.drop(s,axis=1,inplace=True)

df_train=data[data['y'].notnull()]
df_test=data[data['y'].isnull()]

target=df_train['y']
df_train_columns=df_train.columns.tolist()
df_train_columns.remove('ID')
df_train_columns.remove('y')

# data[df_train_columns] = data[df_train_columns].apply(cap)

param = {'num_leaves': 31,
         'min_data_in_leaf': 30,
         'objective':'binary',
         'max_depth': -1,
         'learning_rate': 0.01,
         # "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'auc',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 666}
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=666)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

fold_importance_df = pd.DataFrame()
fold_importance_df["Feature"] = df_train_columns
for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['y'].values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(df_train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=250, early_stopping_rounds = 100)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)


    fold_importance_df["importance_{}".format(fold_)] = clf.feature_importance()
    # fold_importance_df["fold"] = fold_ + 1
    # feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    predictions += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits

from sklearn.metrics import roc_auc_score
print(roc_auc_score(target,oof))

sub=df_test[['ID']]
sub['pred']=predictions
sub.to_csv("submit_"+str(roc_auc_score(target,oof))+".csv",index=False)

print('remove non feature 1')
fold_importance_df['a'] = 0
for i in range(0,10):
    fold_importance_df['a'] += fold_importance_df['importance_{}'.format(i)]
fold_importance_df = fold_importance_df[fold_importance_df['a']!=0]
df_train_columns = list(fold_importance_df['Feature'])


folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=666)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

fold_importance_df = pd.DataFrame()
fold_importance_df["Feature"] = df_train_columns
for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['y'].values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(df_train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=250, early_stopping_rounds = 100)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)


    fold_importance_df["importance_{}".format(fold_)] = clf.feature_importance()
    # fold_importance_df["fold"] = fold_ + 1
    # feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    predictions += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits

from sklearn.metrics import roc_auc_score
print(roc_auc_score(target,oof))

sub=df_test[['ID']]
sub['pred']=predictions
sub.to_csv("submit_"+str(roc_auc_score(target,oof))+".csv",index=False)


print('remove non feature 2')
fold_importance_df['a'] = 0
for i in range(0,10):
    fold_importance_df['a'] += fold_importance_df['importance_{}'.format(i)]

fold_importance_df = fold_importance_df[fold_importance_df['a']>=5]
df_train_columns = list(fold_importance_df['Feature'])

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=666)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

fold_importance_df = pd.DataFrame()
fold_importance_df["Feature"] = df_train_columns
for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['y'].values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(df_train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=250, early_stopping_rounds = 100)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)


    fold_importance_df["importance_{}".format(fold_)] = clf.feature_importance()
    # fold_importance_df["fold"] = fold_ + 1
    # feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    predictions += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits

from sklearn.metrics import roc_auc_score
print(roc_auc_score(target,oof))

sub=df_test[['ID']]
sub['pred']=predictions
sub.to_csv("submit_"+str(roc_auc_score(target,oof))+".csv",index=False)