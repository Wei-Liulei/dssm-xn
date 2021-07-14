# -*- coding: utf-8 -*-
# %% [markdown]
# # import and define some useful functions

# %%
import json
from pathlib import Path
import warnings
import tensorflow as tf
from preprocess import (
    getdata,
    processModel,
    prepareFeas,
    plot_loss,
    splitModel,
    prepareTestInput,
)
from deepmatch.models import DSSM, FM,  YoutubeDNN, MIND, NCF
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC

# wlwldsds
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # 禁用gpu(导入tf前)
warnings.filterwarnings("ignore")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from tensorflow.python.keras.models import Model
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives
# %% [markdown]
# # read data，select useful features
# %% 读取数据, 可能需要的特征
# 预处理需要的数据特征，模型训练时可能不需要所有这些特征
user_feas_raw = ['user_key', 'device_model', 'device_brand', 'app_version',
                 'platform', 'net_type', 'user_channel', 'sex']  # , 'login',  'register_time',
user_feas = user_feas_raw + ['hist_item_id_pad', 'hist_len']
item_feas = ['item_id', 'channel', 'gender', 'content_size', 'collect_cnt',
             'view_cnt', 'pub_days', 'color_id']  # ,   # 'tags', # 'pub_time',
context_feas = ['label', 'bhv_time', 'bhv_hour']
# 特征类型分类
float_num_feas = ['collect_cnt', 'view_cnt']
int_num_feas = ['hist_len', 'content_size']
str_cat_feas = ['user_key', 'device_model', 'device_brand', 'channel',
                'app_version', 'platform', 'net_type', 'user_channel']  # , 'login' bool  #
int_cat_feas = ['gender', 'sex', 'color_id', 'item_id']  #
seqL = 10  # 行为序列截断长度

trainX, trainY, testX, testY, item_profile, user_profile = getdata(
                                                    data_path="../../data_set/data0408/",
                                                    cache_path="cache",
                                                    read_sample_cache=True,
                                                    read_rawdata_cache=True,
                                                    user_num=1000,
                                                    seqL=seqL,
                                                    n_rate=5,
                                                    test_n=1,
                                                    user_feas=user_feas_raw,
                                                    item_feas=item_feas,
                                                    context_feas=context_feas,
                                                )
# %% [markdown]
# # preprocess features with  tf
# by tensorflow.keras.layers.experimental.preprocessing
# %% tf 预处理模型 tensorflow.keras.layers.experimental.preprocessing
trainX_d = {k: np.array(v.tolist()) for k, v in dict(trainX).items() if k != 'hist_item_id'}  #
inputs, processed = processModel(trainX_d, float_num_feas, int_num_feas, str_cat_feas, int_cat_feas, seqL=seqL)
# model_processed = tf.keras.Model(inputs, processed)  # 预处理模型
fea_max_idx = {fea: len(set(trainX_d[fea]))+2 for fea in int_cat_feas+str_cat_feas+['item_id']}  # 每个类别特征编码后的取值个数
# %% [markdown]
# # define model and train
# %% Define DSSM model
embedding_dim = 8  # 类别特征emb维度


def build_model():
    embedding_dim = 8  # 类别特征emb维度
    select_userfeas = ['user_key', 'hist_len']  # , 'hist_len', 'device_model', 'device_brand' 选择除了hist_len和hist_item_id_pad以外的特征
    select_itemfeas = ['item_id', 'collect_cnt']  # , 'channel'
    select_feas = select_userfeas + select_itemfeas
    user_features, item_features = prepareFeas(embedding_dim, seqL, select_feas, user_feas, item_feas, fea_max_idx,
                                               float_num_feas, int_num_feas, int_cat_feas, str_cat_feas, False)
    model_match = DSSM(user_features, item_features,
                       user_dnn_hidden_units=(64, 32),
                       item_dnn_hidden_units=(32, 32),
                       dnn_activation='tanh',
                       dnn_use_bn=False,
                       l2_reg_dnn=0,
                       l2_reg_embedding=1e-6,
                       dnn_dropout=0,
                       seed=1024,
                       metric='cos')
    # model_match = YoutubeDNN(user_features, item_features, num_sampled=5, user_dnn_hidden_units=(64, embedding_dim))
    # Now YoutubeNN only support 1 item feature like item_id
    # model_match = MIND(user_features,item_features,dynamic_k=False,p=1,k_max=2,num_sampled=5,
    # user_dnn_hidden_units=(64, embedding_dim))  # Now MIND only support 1 item feature like item_id
    # model_match = NCF(user_features, item_features, user_gmf_embedding_dim=20,
    #                 item_gmf_embedding_dim=20, user_mlp_embedding_dim=32, item_mlp_embedding_dim=32,
    #                 dnn_hidden_units=[128, 64, 32])

    # model_match = FM(user_features, item_features)  # only sparse features
    # model_processed = tf.keras.Model(inputs, processed)
    model = tf.keras.Model(inputs, model_match(processed))
    model.compile(
                optimizer='rmsprop',  # ,  # 'rmsprop''adagrad',
                loss="binary_crossentropy",
                metrics=[AUC(name='AUC'), BinaryAccuracy(name='Accuracy'),
                         Precision(name='Precision'), Recall(name='Recall')]
                  )  #
    return user_features, item_features, model


user_features, item_features, model = build_model()

checkpoint_dir = "./model/checkpoint/temp/"
# model.load_weights(checkpoint_dir)
# model = tf.keras.models.load_model('./model/loss0.33/')
history = model.fit(
                    trainX_d, trainY,
                    batch_size=2**17,  # 7，10. 256 ,13
                    epochs=2**1,
                    validation_split=0.2,
                    # validation_data=ds_test, #(testX, testY),
                    callbacks=[
                        EarlyStopping(monitor='val_loss', patience=10, verbose=2),
                        # tf.keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1),
                        # tf.keras.callbacks.ModelCheckpoint(save_weights_only=True, verbose=1,
                        #                                    os.path.join(checkpoint_dir, "model_{val_loss:.2f}.hdf5"),
                        #                                     filepath=checkpoint_dir,
                        #                                     save_best_only='True', monitor='val_loss')  # , mode='min'
                               ],
                    workers=4,
                    # use_multiprocessing=True,
                    )

# dfhistory = pd.DataFrame(history.history)
plot_loss(history)
# model.evaluate(testX_d,  testY, verbose=2)  # slow
# 保存模型
model.save_weights(checkpoint_dir)
# model.save('./model/loss0.33/')
# %% [markdown]
# # split model into user and item model
# %% 从训练后的模型拆出物品模型和用户模型
user_model, item_model = splitModel(model, inputs, processed, user_features, item_features)
# models can plot : model_processed, model_match, model, model.layers[-1], user_model, user_model.layers[-1]
# sudo apt-get install graphviz
tf.keras.utils.plot_model(model.layers[-1], show_shapes=True)  # , rankdir="LR"
# # %load_ext tensorboard
# # %tensorboard --logdir logs
# %% save model
model_version = "1"
model_export_path = "./model/"
# !!!, save_traces=False 会减少序列化文件大小和时间，但是需要自定义层定义config方法
user_model.save(Path(model_export_path+'u_emb/'+model_version), save_format="tf")
item_model.save(Path(model_export_path+'i_emb/'+model_version), save_format="tf")

# %% test model
test_user_input, test_item_input = prepareTestInput(testX, user_feas, item_profile)
user_embs = user_model.predict(test_user_input, batch_size=2 ** 12)  #
item_embs = item_model.predict(test_item_input, batch_size=2 ** 12)  #
# %% save all iemb result to json
test_item_input['vector'] = item_embs
item_emb_json = [
    {"item_id": str(i), "gender": j.tolist(), "vector": k.tolist()}
    for i, j, k in zip(
        test_item_input["item_id"], test_item_input["gender"], test_item_input["vector"]
    )
]

with open('item_emb_json.json', 'w+') as f:
    json.dump(item_emb_json, f)
# with open('item_emb_json.json', 'r+') as f:
#     item_emb_json = json.load(f)

# %% load item model and predict first 2 items
load_model = tf.keras.models.load_model(Path(model_export_path+'i_emb/'+model_version))
test_item_input_tmp = {k: v[:2] for k, v in test_item_input.items()}
# test_user_input_tmp = {k: v[:2] for k, v in test_user_input.items()}
embs_tmp = load_model.predict(test_item_input_tmp, batch_size=2 ** 12)
