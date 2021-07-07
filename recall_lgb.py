# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 20:53:55 2021

@author: Administrator
"""

import json
import pickle
import os
import warnings
from pathlib import Path
from tensorflow.python.keras.models import Model
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from wllutils.evaluate import plot2Dscatter
from preprocess import getdata


# %% 读取数据, 可能需要的特征
# data_path = '../data/0225_0303/'  # 原始数据路径
# data_path = '../data/data21_0400_0408/'
data_path = '../data0408/'
cache_path = 'cache'  # 缓存数据路径

seqL = 10  # 行为序列截断长度
# cache_list = 'trainX,trainY,testX,testY,item_profile,user_profile'.split(',')

# for c in cache_list:
#     with open(data_path+f'{cache_path}/'+c+'.pkl', 'rb') as f:
#         exec(f'{c} = pickle.load(f)')
        
# %% copy from dssm_xn.py !!!
user_feas_raw = ['user_key', 'device_model', 'device_brand', 'app_version', 'platform','net_type', 'user_channel', 'sex']  # , 'login',  'register_time',
user_feas = user_feas_raw + ['hist_item_id_pad', 'hist_len']
item_feas = ['item_id', 'channel', 'gender', 'content_size', 'collect_cnt', 'view_cnt', 'pub_days', 'color_id'] # ,   # 'tags', # 'pub_time',
context_feas = ['label', 'bhv_time', 'bhv_hour']
trainX,trainY,testX,testY,item_profile,user_profile = getdata(data_path='../data0408/',
                                                              cache_path='cache',
                                                              read_sample_cache= True,
                                                              read_rawdata_cache=True,                     
                                                              user_num=1000,   
                                                              seqL=seqL,
                                                              n_rate=5,
                                                              test_n=1,
                                                              user_feas=user_feas_raw,
                                                              item_feas=item_feas,
                                                              context_feas=context_feas)
# %%
cate_feas = ['user_key' , 'item_id', 'device_model', 'device_brand', 'app_version', 'platform', 
              'net_type', 'user_channel', 'sex', 'channel', 'gender', 'color_id', 'bhv_hour']  #
train_X = trainX.copy()
train_X.drop(['hist_item_id','hist_item_id_pad'], axis=1, inplace=True)  # , 'item_id'
# 'hist_item_id'
test_X = testX.copy()
test_X.drop(['hist_item_id','hist_item_id_pad'], axis=1, inplace=True)  # , 'item_id'

for c in cate_feas:
    train_X[c] = train_X[c].astype('category')
    test_X[c] = test_X[c].astype('category')

lgb_train = lgb.Dataset(train_X, label=trainY)
lgb_valid = lgb.Dataset(test_X, label=testY, reference=lgb_train)

# ================================================================================
# 参数
# ================================================================================
boost_round = 50
early_stop_rounds = 20
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['auc', 'average_precision'],
    'num_leaves': 100,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'min_data_in_leaf': 20,  # 叶子可能具有的最小记录数	默认20，过拟合时用
    'max_depth': -1,  # 树的最大深度
}
results = {}
# ================================================================================
# 训练
# ================================================================================
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=boost_round,
                valid_sets=(lgb_valid, lgb_train),
                valid_names=('test', 'train'),
                # early_stopping_rounds=early_stop_rounds,
                evals_result=results)

from sklearn.metrics  import accuracy_score, recall_score, roc_auc_score, precision_score
from sklearn.metrics import confusion_matrix,f1_score

# ==============================================================================
# 评估模型
# ================================================================================
is_eval = 1
if is_eval:
    print("start predict")
    y_pred_train = gbm.predict(train_X, num_iteration=gbm.best_iteration)
    y_pred_test = gbm.predict(test_X, num_iteration=gbm.best_iteration)
    # y_pred_val = gbm.predict(val_X, num_iteration=gbm.best_iteration)
    print("end predict")
    threshold = 0.5
    print('train auc: {:.5} '.format(roc_auc_score(trainY, y_pred_train)))
    print('test auc: {:.5} '.format(roc_auc_score(testY, y_pred_test)))

    print('train accuracy: {:.5} '.format(accuracy_score(trainY, y_pred_train > threshold)))
    print('test accuracy: {:.5} '.format(accuracy_score(testY, y_pred_test > threshold)))

    print('train precision: {:.5} '.format(precision_score(trainY, y_pred_train > threshold)))
    print('test precision: {:.5} '.format(precision_score(testY, y_pred_test > threshold)))

    print('train recall_score: {:.5} '.format(recall_score(trainY, y_pred_train > threshold)))
    print('test recall_score: {:.5} '.format(recall_score(testY, y_pred_test > threshold)))

    print('train f1_score: {:.5} '.format(f1_score(trainY, y_pred_train > threshold)))
    print('test f1_score: {:.5} '.format(f1_score(testY, y_pred_test > threshold)))

    c_m = pd.DataFrame(confusion_matrix(trainY, y_pred_train > threshold)/len(trainY),
                        columns=['y_pre=0','y_pre=1'],index=['y=0','y=1'])

    print(c_m)

lgb.plot_metric(results)
lgb.plot_importance(gbm, importance_type="gain")

gain_importance_df = pd.DataFrame(train_X.columns.tolist(), columns=['feature'])
gain_importance_df['importance'] = list(gbm.feature_importance(importance_type="gain"))  # 按每个特征的增益排
gain_importance_df.sort_values(by='importance', ascending=False, inplace=True)

split_importance_df = pd.DataFrame(train_X.columns.tolist(), columns=['feature'])  # 按选择特征的次数排序
split_importance_df['importance'] = list(gbm.feature_importance())
split_importance_df.sort_values(by='importance', ascending=False, inplace=True)
gain_importance_df
split_importance_df

# %%  rely on dssm_xn.py variable
testX_d = {k: np.array(v.tolist()) for k, v in dict(testX).items() if k != 'hist_item_id'}  # 
test_item_input = item_profile.reset_index().to_dict(orient='list')  #
test_item_input = {k: np.array(v) for k, v in test_item_input.items()}
test_user_input = {k: np.array(v.tolist()) for k, v in testX_d.items() if k in user_feas}  # if k != 'hist_item_id'


model_version = "1"
model_export_path = "./model/"
item_model = tf.keras.models.load_model(Path(model_export_path+'i_emb/'+model_version))
user_model = tf.keras.models.load_model(Path(model_export_path+'u_emb/'+model_version))

user_embs = user_model.predict(test_user_input, batch_size=2 ** 12)  #
item_embs = item_model.predict(test_item_input, batch_size=2 ** 12)  #


# pip install faiss-cpu
import numpy as np
import faiss
from tqdm import tqdm

def recall_N(y_true, y_pred, N=50):
    return len(set(y_pred[:N]) & set(y_true)) * 1.0 / len(y_true)

# test_true_label = {line[0]: [line[2]] for line in test_set}
test_df = testX[['user_key', 'item_id']]
test_df['label'] = testY
# test_df = trainX[['user_key', 'item_id']]
# test_df['label'] = trainY

test_df = test_df[test_df['label']==1]
test_df = test_df[['user_key', 'item_id']]
true_df = pd.DataFrame(test_df.groupby('user_key')['item_id'].apply(list)) #.reset_index()
# true_df = true_df[true_df['item_id'].apply(len)>=5]

# test_user_input = user_profile.reset_index()[user_profile.reset_index()['user_key'].isin(set(true_df.index))].to_dict(orient='list') # 
# test_user_input = {k: np.array(v) for k, v in test_user_input.items() if k!='index'}
# user_embs = user_model.predict(test_user_input, batch_size=2 ** 12)  #


# index = faiss.IndexFlatIP(embedding_dim)
index = faiss.IndexFlatIP(32)
faiss.normalize_L2(item_embs)
index.add(item_embs)
faiss.normalize_L2(user_embs)
D, I = index.search(user_embs, 20)
s = []
hit = 0
for i, uid in tqdm(enumerate(true_df.index)):
    pred = [item_profile.index.values[x] for x in I[i]]
    recall_score = recall_N(list(true_df.loc[uid]['item_id']), pred, N=200)
    s.append(recall_score)
print("recall", np.mean(s))
# print("hr", hit / len(true_df))


# %%
# from ge_classify import  Classifier # read_node_label,
# def evaluate_embeddings(embeddings, lable_path):
#     X, Y = read_node_label(lable_path)   # skip_head=True
#     key, label =
#     tr_frac = 0.8
#     print("Training classifier using {:.2f}% nodes...".format(tr_frac * 100))
#     clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
#     clf.split_train_evaluate(X, Y, tr_frac)

# with open('item_emb_json.json', 'r+') as f:
#     item_emb_json = json.load(f)
# item_emb_json_df = pd.DataFrame(item_emb_json)


# from sklearn.manifold import TSNE
# model_tsne = TSNE(n_components=2)
# node_pos = model_tsne.fit_transform(np.array(item_emb_json_df['vector'].tolist()))
# side_info_df = pd.DataFrame(item_profile.reset_index())  # .set_index()
# side_info_df.columns = ['item_id'] + ['fea' + str(i) for i in range(1, side_info_df.shape[1])]
# node_pos_df = pd.DataFrame(node_pos, columns=('eb0', 'eb1')).reset_index().rename(columns={'index': 'item_id'})
# item_df = pd.merge(side_info_df, node_pos_df, on='item_id')

# # for c in [1, 2]:  # , 3 第1，2，3列
# #     plot2Dscatter(item_df, label_idx=c, label_num=5)

# def plot2Dscatter(data_df, isPlotConvexhull=True, labelValues='all'):
#     """
#     data_df : pandas DataFrame.
#         The first three columns are X,Y,Z and the fourth column is label
#     """
#     if labelValues == 'all':
#         labelValues = set(data_df.iloc[:, -1])
#     colors = [plt.cm.tab10(i/float(len(labelValues)-1)) for i in range(len(labelValues))]
#     # fig = plt.figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')  # 分辨率，背景颜色，边缘颜色
#     for i, v in enumerate(labelValues):
#         points_df = data_df[data_df.iloc[:, -1] == v]
#         plt.plot(points_df.iloc[:, 0], points_df.iloc[:, 1], 'o', c=colors[i], label=v)  # ,
#         plt.legend()
#         if isPlotConvexhull:
#             points = points_df.iloc[:, :2].values
#             hull = ConvexHull(points)
#             for simplex in hull.simplices:
#                 plt.plot(points[simplex, 0], points[simplex, 1], c=colors[i])  # , points0[simplex, 2]




# # data = read_part_csv('../data/data21_0400_0408/data/2021-04-03/dataCSV1dt16/' + '*.csv', sep='$')
# # data1 = read_part_csv('../data/data21_0400_0408/data/2021-04-03/dataCSV1dt16day/' + '*.csv', sep='$')
# # # E:/workplace/data/data21_0400_0408/data/2021-04-06
# # # data.head()

# # data.label.value_counts()
