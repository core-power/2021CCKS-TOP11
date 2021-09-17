import json
import os
print('因果关系模型推理结果正在聚合')
with open('dense_pred.txt','r',encoding='utf-8') as f:
    dense_pred = f.readlines()
f.close()

with open('ner_pred.txt','r',encoding='utf-8') as f:
    ner_pred = f.readlines()
f.close()
print('dense_pred:',len(dense_pred))
print('ner_pred',len(ner_pred))
t = open('彭于晏岳阳分晏_valid_result.txt','w',encoding='utf-8')
for i in range(len(ner_pred)):
    l1 = json.loads(ner_pred[i])
    l2 = json.loads(dense_pred[i])
    result = {**l1['result'][0], **l2['result'][0]}
    result_json = json.dumps({'text_id': str(l1['text_id']),'text':l1['text'],'result': [result]},ensure_ascii=False)
    t.write(result_json+'\n')
t.close()
os.remove("dense_pred.txt")
os.remove("ner_pred.txt")
print('模型结果写入完毕')
