# 2021年CCKS因果抽取比赛TOP11比赛分享


#训练代码


#ner模型训练代码

ner_gp.py

#原因标签分类模型

bert_dense_v1.py


#结果标签分类模型

bert_dense_v2.py

#实体候选集排序模型

sort_model.py

##推理代码


#因果标签推理代码

dense_inference.py

#因果实体推理以及排序候选集模型推理

ner_inference.py

###因果标签推理结果和因果实体推理结果聚合程序

aggregate_result.py

###统一执行推理sh文件

bash inference.sh
