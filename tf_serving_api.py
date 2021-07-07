# -*- coding: utf-8 -*-
"""
tf serving 
"""

# =============================================================================
# # # 单模型 docker
# =============================================================================
# sudo docker run -t --rm -p 8500:8500 -p 8501:8501 \
#     -v "/home/weiliulei/deepmatch/model/u_emb:/models/dssm_samh" \
#     -e MODEL_NAME=dssm_samh \
#     tensorflow/serving &

# curl get http://localhost:8501/v1/models/dssm_samh/metadata
# curl -d '{"instances": [65984, 38568]}' -X POST http://localhost:8501/v1/models/dssm_samh: predict


# =============================================================================
# # # 多模型 docker
# =============================================================================

sudo docker run -d -t --rm -p 8500:8500 -p 8501:8501 \
    -v "/media/liulei/107F19C9107F19C9/xndm/tfserv/model:/models/" tensorflow/serving \
    --model_config_file=/models/models.config \
    --model_config_file_poll_wait_seconds=60

# =============================================================================
# saved_model_cli
# =============================================================================
saved_model_cli show --dir ./1/* --all

# =============================================================================
# # rest api 查看元数据和预测 curl
# =============================================================================

curl get http://localhost:8501/v1/models/query_embedding-samh-comic-73/
curl get http://localhost:8501/v1/models/query_embedding-samh-comic-73/versions/1/metadata

curl -d '{"instances": [{"user_key": ["669bce1f9c9922cc", "25976860b7906282"], "device_model":["vivo X21UD A","V1928A"],
"device_brand":["vivo", "vivo"]}]}' -X POST http://localhost:8501/v1/models/query_embedding-samh-comic-modelid:predict

curl -d '{"instances": [{"item_id": [108535, 200003], "channel":["恋爱", "穿越"], "gender":[2, 1],
"collect_cnt":[17817, 559052], "view_cnt": [ 10297295, 849733739]}]}' -X POST http://localhost:8501/v1/models/item_embedding-samh-comic-modelid:predict


# =============================================================================
# # python rest
# =============================================================================
import requests
import json
import pprint as pp

user_model_name = 'query_embedding-samh-comic-73'
user_data = {"instances": [{'user_key': ['669bce1f9c9922cc', '25976860b7906282'],
                            'device_model': ['vivo X21UD A', 'V1928A'],
                            'device_brand': ['vivo', 'vivo']}]}

item_model_name = 'item_embedding-samh-comic-73'
item_data = {"instances": [{'item_id': [108535, 200003],
                            'channel': ['恋爱', '穿越'],
                            'gender': [2, 1],
                            'collect_cnt': [17817, 559052],
                            'view_cnt': [10297295, 849733739]}]}

for name, data in [[user_model_name, user_data], [item_model_name, item_data]][:1]:
    r_metadata = requests.get(f'http://localhost:8501/v1/models/{name}/metadata')
    pp.pprint(r_metadata.json())
    r = requests.post(f'http://localhost:8501/v1/models/{name}:predict', data=json.dumps(data))
    pp.pprint(r.json())


# =============================================================================
# # grpc api
# =============================================================================
./grpcurl -plaintext -d '{"user_id":"0","device_id":"869984030446699","platform":"samh","item_types":["comic"],
"features":["user_key","user_channel","sex"]}' 127.0.0.1:5004 proto.AIRecProfileUserServe/QueryUserInfosOnFeatures

./grpcurl -plaintext -d '{"item_id":"1000",
"features":["user_read_dense_n7d_dist_channel_score_criterion"]}' 127.0.0.1:5004 proto.AIRecProfileItemServe/QueryUserInfosOnFeatures

./grpcurl -plaintext get localhost:8501 proto.GetModelMetadataRequest 


model_config_list:{
    config:{
        name:"query_embedding-samh-comic-modelid"
        base_path:"/models/samh/model/recall/queryembedding/query_embedding-samh-comic-modelid/"
        model_platform:"tensorflow"
        model_version_policy {
            specific {
            versions: 1
            }
        }
    },
    config:{
        name:"item_embedding-samh-comic-modelid"
        base_path:"/models/samh/model/recall/item_embedding/item_embedding-samh-comic-modelid/"
        model_platform:"tensorflow"
        model_version_policy {
            specific {
            versions: 1
            }
        }
    },
}
