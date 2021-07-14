import os
import random
import numpy as np
from tqdm import tqdm
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import time
from glob import glob
# from .evaluate import timmer
import warnings
warnings.filterwarnings("ignore")
import numpy
import pickle
import tensorflow as tf
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat
from tensorflow.python.keras.models import Model
from deepmatch.models import DSSM
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Precision, Recall, AUC


def timmer(func):
    def deco(*args, **kwargs):
        start_time = time.time()
        print(f'\n{time.strftime("%H:%M:%S", time.localtime())} {func.__name__} start running ...')
        res = func(*args, **kwargs)
        end_time = time.time()
        print(f'{time.strftime("%H:%M:%S", time.localtime())} {func.__name__} costed {(end_time-start_time):.2f} Sec')
        return res
    return deco


@timmer
def read_part_csv(file_path, parse_dates=None, sep=','):
    csv_files = sorted(glob(file_path))
    frames = (pd.read_csv(file, parse_dates=parse_dates, sep=sep) for file in tqdm(csv_files))
    concat_df = pd.concat(frames)
    concat_df.reset_index(drop=True, inplace=True)
    print('read data done ...')
    return concat_df


@timmer
def get_profile(data, item_feas, user_feas, user_num=0):  #
    '''
    对用户采样，数值特征归一化，提取用户画像和物品画像
    '''
    item_profile = data[item_feas].drop_duplicates('item_id')
    item_profile.set_index("item_id", inplace=True)
    user_profile = data[user_feas].drop_duplicates('user_key')
    user_profile.set_index('user_key', inplace=True)
    # if user_num:
    #     data = data[data['user_key'].isin(data['user_key'].value_counts()[:user_num].index)]    
    return item_profile, user_profile  # , fea_max_idx


def reduce_mem_usage_num(df, verbose=True, deep=True, except_cols=()):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=deep).sum() / 1024**2
#     for col in df.columns:
    for col in [col for col in df.columns if col not in except_cols]:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)    # ！！！
#                     df[col] = df[col].astype(np.float32)    ##  np.float16 在pandas里是没有的。在这改成32
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=deep).sum() / 1024**2
    if verbose:
        print(f'{start_mem:.2f} Mb =>> {end_mem:.2f} Mb  compression ratio: {end_mem/start_mem:.2f}')
    return df


def reduce_mem_usage_cate(df, verbose=True, deep=True, except_cols=()):
    start_mem = df.memory_usage(deep=deep).sum() / 1024**2
    for c in [col for col in df.columns if col not in except_cols]:
        # col_type = df[c].dtypes
        mem_before = df[c].memory_usage(deep=True) / 1e6  #
        mem_cate = df[c].astype('category').memory_usage(deep=True) / 1e6  #
        c_type = df[c].dtype
        compress_rate = mem_before/mem_cate
        if compress_rate > 1:
            print(c, ' ', c_type, ':', f'{compress_rate:.2f}')
            df[c] = df[c].astype('category')
    end_mem = df.memory_usage(deep=deep).sum() / 1024**2
    if verbose:
        print('='*20)
        print(f'{start_mem:.2f} Mb =>> {end_mem:.2f} Mb  compression ratio: {end_mem/start_mem:.2f}')
    return df


def read_data(data_path, cache_path,  read_cache=True):
    # 压缩数据并保存为 pickle提升加载速度
    pkl_path = data_path + f'{cache_path}/' + 'mem_reduced_data.pkl'
    print(pkl_path)
    if os.path.exists(pkl_path) and read_cache:
        data = pd.read_pickle(pkl_path)
        print('mem_reduced_data exist:')
    else:
        print('%n read raw data:')
        data = read_part_csv(data_path + '*.csv', sep='$')
        data = data[data.label == 1]  # 只需要正样本
        except_cols = [c for c in data.columns if c.endswith('time')] + ['day']
        except_cols += ['collect_cnt', 'view_cnt']
        data = reduce_mem_usage_num(data)
        data = reduce_mem_usage_cate(data, except_cols=except_cols)
        # context feas
        data = p2rDays(data)
        # data.dropna(1, inplace=True, how='all')
        data.to_pickle(pkl_path)
    return data



# # need globals namespace so this function can't put in other py files 
# def save_cache_data(data_path, cache_path, cache_list):
#     for c in cache_list:
#         with open(data_path+f'/{cache_path}/'+c+'.pkl', 'wb') as f:
#             pickle.dump(eval(c), f)

# # can't define global value in funciton
# def load_cache_data(data_path, cache_path, cache_list):
#     for c in cache_list:
#         with open(data_path+f'{cache_path}/'+c+'.pkl', 'rb') as f:
#             exec(f'{c} = pickle.load(f)')


def getdata(data_path = '../data0408/', 
            cache_path = 'cache', 
            read_sample_cache=False,  # 是否重新采样，若否则读缓存后的采样数据
            user_num=1000,   # 若重采样，采样活跃用户数
            read_rawdata_cache=True,  # 若重新采样，是否读取采样前的缓存数据
            n_rate=5, # 采样负样本相对正样本的倍数
            test_n=1,  # 每个用户测试样本个数
            user_feas=[],  # 用户特征列表
            item_feas=[],  # 物品特征列表
            context_feas=[],  # 上下文特征列表 
            seqL=10 # 采样行为序列长度
            ):

    cache_list = 'trainX,trainY,testX,testY,item_profile,user_profile'.split(',')
    if not read_sample_cache:
        # user_num = 1000  # 0 : 所有用户
        # n_rate = 5  # 采样负样本相对正样本的倍数
        # test_n = 1  # 每个用户测试样本个数
        data = read_data(data_path, cache_path, read_cache=read_rawdata_cache)  # 读取原始训练数据
        item_profile, user_profile = get_profile(data, item_feas, user_feas, user_num=user_num)
        if user_num:
            data = data[data['user_key'].isin(data['user_key'].value_counts()[:user_num].index)]    
        data['sex'] = data['sex'].fillna(2.0).astype('int')
        # add_feas = tuple([c for c in user_feas if c != 'label']) + tuple([c for c in user_feas if c != 'user_key']) + tuple([c for c in item_feas if c != 'item_id'])
        add_feas = list(filter(lambda x: x not in ['label', 'user_key', 'item_id'], sum([user_feas, context_feas], [])))  # , item_feas !!!
        # for c in ['collect_cnt', 'view_cnt']:  # TODO 改成tf模型内实现
        #     data[c] = (data[c] + 1).apply(np.log)
        trainX, trainY, testX, testY = gen_data_set(data, seqL=seqL, n_rate=n_rate, test_n=test_n, add_feas=add_feas, is_shuffle=True)
        trainX = pd.merge(trainX, item_profile.reset_index()[item_feas], on='item_id', how='left')
        testX = pd.merge(testX, item_profile.reset_index()[item_feas], on='item_id', how='left')    
        # trainX.join(item_profile.reset_index()[item_feas], on='item_id')
        print(len(trainX)/len(testX))
        # save_cache_data(data_path, cache_path, cache_list)
        for c in cache_list:
            with open(data_path+f'/{cache_path}/'+c+'.pkl', 'wb') as f:
                pickle.dump(eval(c), f)
    else:
        # load_cache_data(data_path, cache_path, cache_list) 
        # for c in cache_list:
        #     with open(data_path+f'{cache_path}/'+c+'.pkl', 'rb') as f:
        #         exec(f'{c} = pickle.load(f)')
        # 不想复制粘贴，但是 exec无法复制非局域变量。。
        with open(data_path+f'{cache_path}/'+'trainX'+'.pkl', 'rb') as f:
            trainX = pickle.load(f)
        with open(data_path+f'{cache_path}/'+'trainY'+'.pkl', 'rb') as f:
            trainY = pickle.load(f)
        with open(data_path+f'{cache_path}/'+'testX'+'.pkl', 'rb') as f:
            testX = pickle.load(f)
        with open(data_path+f'{cache_path}/'+'testY'+'.pkl', 'rb') as f:
            testY = pickle.load(f)
        with open(data_path+f'{cache_path}/'+'item_profile'+'.pkl', 'rb') as f:
            item_profile = pickle.load(f)
        with open(data_path+f'{cache_path}/'+'user_profile'+'.pkl', 'rb') as f:
            user_profile = pickle.load(f)            
    return trainX,trainY,testX,testY,item_profile,user_profile


def get_normalization_layer(name, dataset):
    normalizer = preprocessing.Normalization()
    normalizer.adapt(dataset[name])
    return normalizer


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    if dtype == 'string':
        index = preprocessing.StringLookup(max_tokens=max_tokens)
    else:
        index = preprocessing.IntegerLookup(max_values=max_tokens)
    index.adapt(dataset[name])
    return index


def processModel(trainX_d, 
                 float_num_feas=(), 
                 int_num_feas=(), 
                 str_cat_feas=(), 
                 int_cat_feas=(), 
                 seqL=10):
    # float_num_feas = ['collect_cnt', 'view_cnt']
    # int_num_feas = ['hist_len', 'content_size']
    # str_cat_feas = ['user_key', 'device_model', 'device_brand', 'channel', 'app_version', 'platform',
    #                 'net_type', 'user_channel']  # , 'login' bool  #
    # int_cat_feas = ['gender', 'sex', 'color_id', 'item_id']  #
    inputs = {}  # 指定输入特征的 type，shape，name
    processed = {}  # 数值特征归一化或类别特征标签编码后的特征
    for fea in float_num_feas:
        inputs[fea] = tf.keras.Input(shape=(1,), name=fea, dtype='float64')  # 定义输入特征shape， name， type
        normalization_layer = get_normalization_layer(fea, trainX_d)  # 用训练数据 adapt
        processed[fea] = normalization_layer(inputs[fea])  # transform
    for fea in int_num_feas:
        inputs[fea] = tf.keras.Input(shape=(1,), name=fea, dtype='int64')  # 定义输入特征shape， name， type
        normalization_layer = get_normalization_layer(fea, trainX_d)  # 用训练数据 adapt
        processed[fea] = normalization_layer(inputs[fea])  # transform
    for fea in str_cat_feas:
        inputs[fea] = tf.keras.Input(shape=(1,), name=fea, dtype='string')
        encoding_layer = get_category_encoding_layer(fea, trainX_d, dtype='string', max_tokens=None)
        processed[fea] = encoding_layer(inputs[fea])
    for fea in int_cat_feas:
        inputs[fea] = tf.keras.Input(shape=(1,), name=fea, dtype='int64')
        encoding_layer = get_category_encoding_layer(fea, trainX_d, dtype='int64', max_tokens=None)
        processed[fea] = encoding_layer(inputs[fea])
    # 行为序列特征, 最后一个处理特征要求是item_id
    seq_fea = 'hist_item_id_pad'
    inputs[seq_fea] = tf.keras.Input(shape=(seqL,), name=seq_fea, dtype='int64')
    processed[seq_fea] = encoding_layer(inputs[seq_fea])
    return inputs, processed


def splitModel(model, inputs, processed, user_features, item_features):
    user_inputs = {k: v for k, v in inputs.items() if k in [c.name for c in user_features]}
    user_processed = {k: v for k, v in processed.items() if k in [c.name for c in user_features]}
    item_inputs = {k: v for k, v in inputs.items() if k in [c.name for c in item_features]}
    item_processed = {k: v for k, v in processed.items() if k in [c.name for c in item_features]}
    
    user_model_proessed = Model(inputs=model.layers[-1].user_input, outputs=model.layers[-1].user_embedding)
    item_model_proessed = Model(inputs=model.layers[-1].item_input, outputs=model.layers[-1].item_embedding)
    user_model = Model(user_inputs,  user_model_proessed(user_processed))
    item_model = Model(item_inputs,  item_model_proessed(item_processed))
    return user_model, item_model


#  Generate user features for testing and full item features for retrieval
def prepareTestInput(testX, user_feas, item_profile):
    testX_d = {k: np.array(v.tolist()) for k, v in dict(testX).items() if k != 'hist_item_id'}  # 
    test_user_input = {k: np.array(v.tolist()) for k, v in testX_d.items() if k in user_feas}  # if k != 'hist_item_id'
    test_item_input = item_profile.reset_index().to_dict(orient='list')  #
    test_item_input = {k: np.array(v) for k, v in test_item_input.items()}
    return test_user_input, test_item_input


def prepareFeas(embedding_dim, 
                seqL, 
                select_feas, 
                user_feas,
                item_feas, 
                fea_max_idx, 
                float_num_feas, 
                int_num_feas,
                int_cat_feas, 
                str_cat_feas, 
                is_addhis):
    num_feas = float_num_feas + int_num_feas
    cat_feas = int_cat_feas + str_cat_feas
    user_features = [SparseFeat(c, fea_max_idx[c], embedding_dim) for c in set(select_feas)&set(user_feas)&set(cat_feas)]  # 'device_model', 'net_type',  'platform', 'sex'  set(user_feas)&set(cat_feas)
    user_features += [DenseFeat(c, 1) for c in set(select_feas)&set(user_feas)&set(num_feas)]
    item_features = [SparseFeat(c, fea_max_idx[c], embedding_dim) for c in set(select_feas)&set(item_feas)&set(cat_feas)]  # , 'color_id' ['item_id', 'channel', 'gender', 'color_id']
    item_features += [DenseFeat(c, 1,) for c in set(select_feas)&set(item_feas)&set(num_feas)]  # ,  , 'pub_days'
    if is_addhis:    
        user_features += [VarLenSparseFeat(SparseFeat('hist_item_id_pad', fea_max_idx['item_id'], embedding_dim, embedding_name="item_id"),
                                            seqL, 'mean', 'hist_len')]    
        user_features += [DenseFeat('hist_len', 1)]  # DenseFeat(hist_len) 必须在 VarLenSparseFeat(hist_item_id_pad) 后面    
    return user_features, item_features


def plot_loss(history):
    dfhistory = pd.DataFrame(history.history)
    print(dfhistory)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def transdate(int_time):
    if pd.isnull(int_time):
        return int_time
    return time.strftime("%Y-%m-%d", time.localtime(int(int_time)))  # %H:%M:%S


def p2rDays(df):
    for t in ["pub_time", "register_time"]:
        # .value_counts()#.shape
        df[t] = pd.to_datetime(df[t].apply(transdate))
    df['pub_days'] = (pd.to_datetime(df['day'], format='%Y-%m-%d') - df['pub_time']).dt.days  # %H:%M:%S'
    df['reg_days'] = (pd.to_datetime(df['day'], format='%Y-%m-%d') - df['register_time']).dt.days  # ,format='%Y-%m-%d'
    df['bhv_hour'] = pd.to_datetime(df.bhv_time, unit='s').dt.hour
    return df


def pad_and_get_label(train_set, seq_max_len, contexts=()):
    columns = ['user_key', 'item_id', 'label', 'hist_item_id', 'hist_len'] + list(contexts)
    context = pd.DataFrame(train_set, columns=columns)
    context['hist_item_id_pad'] = pad_sequences(context['hist_item_id'], maxlen=seq_max_len, padding='post', truncating='post', value=0).tolist()
    train_label = context.pop('label')
    # return {k: np.array(v.tolist()) for k, v in dict(context).items() if k != 'hist_item_id'}, train_label  #.values
    return context, train_label


def gen_data_set(data, add_feas=(), n_rate=0, test_n=1, seqL=50, is_shuffle=True):
    '''
    采样正负样本，训练集和测试集
    负样本为未点击物品随机采样
    测试机为每个用户最后test_n次行为
    '''
    # print(data.user_key.nunique())
    userCol, itemCol, timeCol = 'user_key', 'item_id', 'bhv_time'
    data.sort_values(timeCol, inplace=True)
    item_ids = data[itemCol].unique()

    train_set = []
    test_set = []

    # print(len(data.groupby(userCol)))
    for user_id, user_df in tqdm(data.groupby(userCol, observed=True)):
        pos_list = user_df[itemCol].tolist()  #
        context_list = [user_df[c].tolist() for c in add_feas]  # 从未点击物品中随机采样生成负样本
        candidate_set = list(set(item_ids) - set(pos_list))
        neg_list = np.random.choice(candidate_set, size=len(pos_list)*n_rate, replace=True)

        for i in range(len(pos_list)-test_n):   # 每个用户的点击序列除去最后 test_n个点击行为，采样正负样本得到训练集
            # context_feas = [c[i] for c in context_list]  # 上下文特征
            hist = pos_list[:i]
            # user_id, his_item, pos_id,  1, his_item_len
            train_set.append((user_id, pos_list[i], 1, hist[::-1], len(hist[::-1])) + tuple(c[i] for c in context_list))  # , rating_list[i]
            for negi in range(n_rate):
                # user_id, his_item, neg_id,  0, his_item_len
                train_set.append((user_id, neg_list[i*n_rate+negi], 0, hist[::-1], len(hist[::-1])) + tuple(c[i] for c in context_list))

        for i in range(max(len(pos_list) - test_n, 0), len(pos_list)):  # 每个用户最后test_n个行为当作测试集，若行为数少于 test_n 则都放入测试集
            hist = pos_list[:i]
            # user_id, his_item, pos_id,  1, his_item_len
            test_set.append((user_id, pos_list[i], 1, hist[::-1], len(hist[::-1])) + tuple(c[i] for c in context_list))  # , rating_list[i]
            for negi in range(n_rate):
                # user_id, his_item, neg_id,  0, his_item_len
                test_set.append((user_id, neg_list[i*n_rate+negi], 0, hist[::-1], len(hist[::-1])) + tuple(c[i] for c in context_list))

    if is_shuffle:  
        random.shuffle(train_set)
        random.shuffle(test_set)
    trainX, trainY = pad_and_get_label(train_set, seqL, contexts=add_feas)
    testX, testY = pad_and_get_label(test_set, seqL, contexts=add_feas)
    return trainX, trainY, testX, testY



# def gen_data_set_sdm(data, seq_short_len=5, seq_prefer_len=50):

#     data.sort_values("timestamp", inplace=True)
#     train_set = []
#     test_set = []
#     for reviewerID, hist in tqdm(data.groupby('user_id')):
#         pos_list = hist['movie_id'].tolist()
#         genres_list = hist['genres'].tolist()
#         rating_list = hist['rating'].tolist()
#         for i in range(1, len(pos_list)):
#             hist = pos_list[:i]
#             genres_hist = genres_list[:i]
#             if i <= seq_short_len and i != len(pos_list) - 1:
#                 train_set.append((reviewerID, hist[::-1], [0]*seq_prefer_len, pos_list[i], 1, len(hist[::-1]), 0,
#                                   rating_list[i], genres_hist[::-1], [0]*seq_prefer_len))
#             elif i != len(pos_list) - 1:
#                 train_set.append((reviewerID, hist[::-1][:seq_short_len], hist[::-1][seq_short_len:], pos_list[i], 1, seq_short_len,
#                 len(hist[::-1])-seq_short_len, rating_list[i], genres_hist[::-1][:seq_short_len], genres_hist[::-1][seq_short_len:]))
#             elif i <= seq_short_len and i == len(pos_list) - 1:
#                 test_set.append((reviewerID, hist[::-1], [0] * seq_prefer_len, pos_list[i], 1, len(hist[::-1]), 0,
#                                   rating_list[i], genres_hist[::-1], [0]*seq_prefer_len))
#             else:
#                 test_set.append((reviewerID, hist[::-1][:seq_short_len], hist[::-1][seq_short_len:], pos_list[i], 1, seq_short_len,
#                 len(hist[::-1])-seq_short_len, rating_list[i], genres_hist[::-1][:seq_short_len], genres_hist[::-1][seq_short_len:]))

#     random.shuffle(train_set)
#     random.shuffle(test_set)

#     print(len(train_set[0]), len(test_set[0]))

#     return train_set, test_set


# def gen_model_input_sdm(train_set, user_profile, seq_short_len, seq_prefer_len):

#     train_uid = np.array([line[0] for line in train_set])
#     short_train_seq = [line[1] for line in train_set]
#     prefer_train_seq = [line[2] for line in train_set]
#     train_iid = np.array([line[3] for line in train_set])
#     train_label = np.array([line[4] for line in train_set])
#     train_short_len = np.array([line[5] for line in train_set])
#     train_prefer_len = np.array([line[6] for line in train_set])
#     short_train_seq_genres = np.array([line[8] for line in train_set])
#     prefer_train_seq_genres = np.array([line[9] for line in train_set])

#     train_short_item_pad = pad_sequences(short_train_seq, maxlen=seq_short_len, padding='post', truncating='post',
#                                         value=0)
#     train_prefer_item_pad = pad_sequences(prefer_train_seq, maxlen=seq_prefer_len, padding='post', truncating='post',
#                                          value=0)
#     train_short_genres_pad = pad_sequences(short_train_seq_genres, maxlen=seq_short_len, padding='post', truncating='post',
#                                         value=0)
#     train_prefer_genres_pad = pad_sequences(prefer_train_seq_genres, maxlen=seq_prefer_len, padding='post', truncating='post',
#                                         value=0)

#     train_model_input = {"user_id": train_uid, "movie_id": train_iid, "short_movie_id": train_short_item_pad,
#         "prefer_movie_id": train_prefer_item_pad, "prefer_sess_length": train_prefer_len, "short_sess_length":
#         train_short_len, 'short_genres': train_short_genres_pad, 'prefer_genres': train_prefer_genres_pad}

#     for key in ["gender", "age", "occupation", "zip"]:
#         train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

#     return train_model_input, train_label
