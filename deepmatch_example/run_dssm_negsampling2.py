import pandas as pd
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat
from preprocess import gen_data_set, gen_model_input
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Model
from deepmatch.models import DSSM
import tensorflow as tf

data = pd.read_csv("./movielens_sample.txt")

SEQ_LEN = 50
negsample = 3
# %%1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`
features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
feature_max_idx = {}
for feature in features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature]) + 1
    feature_max_idx[feature] = data[feature].max() + 1

user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')
item_profile = data[["movie_id"]].drop_duplicates('movie_id')

user_profile.set_index("user_id", inplace=True)
user_item_list = data.groupby("user_id")['movie_id'].apply(list)
user_item_list.apply(len).sum()


# process sequence features with `gen_date_set` and `gen_model_input`
train_set, test_set = gen_data_set(data, negsample)




# # # # 908/4
# import numpy as np
# from tqdm import tqdm
# data.sort_values("timestamp", inplace=True)
# item_ids = data['movie_id'].unique()

# train_set = []
# test_set = []
# train_len = 0
# for reviewerID, hist in tqdm(data.groupby('user_id')):
#     pos_list = hist['movie_id'].tolist()
#     rating_list = hist['rating'].tolist()

#     if negsample > 0:
#         candidate_set = list(set(item_ids) - set(pos_list))
#         neg_list = np.random.choice(candidate_set, size=len(pos_list)*negsample, replace=True)
#     for i in range(1, len(pos_list)):
#         hist = pos_list[:i]
#         train_len += 1
#         if i != len(pos_list) - 1:  #
#             train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))
#             for negi in range(negsample):
#                 train_set.append((reviewerID, hist[::-1], neg_list[i*negsample+negi], 0, len(hist[::-1])))
#         else:
#             test_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]))


train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)  # 截断或者补0
test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)


# %%2.count # unique features for each sparse field and generate feature config for sequence feature
embedding_dim = 8

user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                        SparseFeat("gender", feature_max_idx['gender'], embedding_dim),
                        SparseFeat("age", feature_max_idx['age'], embedding_dim),
                        SparseFeat("occupation", feature_max_idx['occupation'], embedding_dim),
                        SparseFeat("zip", feature_max_idx['zip'], embedding_dim),
                        VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                    embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                        # DenseFeat('hist_len', 1)
                        ]

item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]

# %%3.Define Model and train
model = DSSM(user_feature_columns, item_feature_columns)  # FM(user_feature_columns, item_feature_columns)
# model.summary()
model.compile(optimizer='adagrad', loss="binary_crossentropy", metrics=['accuracy'])
tf.keras.utils.plot_model(model, show_shapes=True)  # , to_file='DSSM_model.png'


history = model.fit(train_model_input, train_label, batch_size=256, epochs=1, verbose=1, validation_split=0.0,)

# %% 4. Generate user features for testing and full item features for retrieval
test_user_model_input = test_model_input
all_item_model_input = {"movie_id": item_profile['movie_id'].values}

user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)  #
item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)  #

print(user_embs.shape)
print(item_embs.shape)
tf.keras.utils.plot_model(model, show_shapes=True)  #, rankdir="LR"
# %% 5. [Optional] ANN search by faiss  and evaluate the result

test_true_label = {line[0]: [line[2]] for line in test_set}

import numpy as np
import faiss
from tqdm import tqdm
from deepmatch.utils import recall_N

# index = faiss.IndexFlatIP(embedding_dim)
index = faiss.IndexFlatIP(32)
faiss.normalize_L2(item_embs)
index.add(item_embs)
# faiss.normalize_L2(user_embs)


D, I = index.search(user_embs, 50)
s = []
hit = 0
for i, uid in tqdm(enumerate(test_user_model_input['user_id'])):
    try:
        pred = [item_profile['movie_id'].values[x] for x in I[i]]
        # filter_item = None
        recall_score = recall_N(test_true_label[uid], pred, N=50)
        s.append(recall_score)
        if test_true_label[uid] in pred:
            hit += 1
    except:
        print(i)
print("recall", np.mean(s))
print("hr", hit / len(test_user_model_input['user_id']))
