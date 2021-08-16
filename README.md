# dssm-xn

小牛动漫双塔召回模型

已实现微软dssm模型，example有其他双塔召回模型）但是物品侧特征只支持item_id一个特征）

## 代码结构

dssm_xn.py ：dssm model (a two tower model recall model of Microsoft) for xndm 

preprocess.py:  utils code

model : user model and item model from the two tower model

deepmatch_example:  other recall model example from deepmatch

## 依赖包

https://deepmatch.readthedocs.io/en/latest/
pip install deepctr
pip install deepmatch

tensorflow>=2.3
tensorflow.keras.layers.experimental.preprocessing （2.3以上版本才有）

## 建议

安装jupytext
https://github.com/mwouts/jupytext

此jupyter插件使jupyter，可以将ipynb与md、py相互绑定。可直接以jupyter形式打开md和py文件，增强可读性





## 特性

已实现将类别特征标签编码和数值特征归一化。

## todo：

在tf模型预处理中实现特征缩放（取对数）

在tf模型中实现特征缩放（取对数）
