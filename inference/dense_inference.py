import json
from tqdm import tqdm
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.backend import multilabel_categorical_crossentropy
from bert4keras.layers import GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.models import Model
import tensorflow as tf
from bert4keras.backend import keras, K, is_tf_keras
from bert4keras.backend import sequence_masking
from bert4keras.backend import recompute_grad
from keras import initializers, activations
from keras.layers import *
from keras.layers import Lambda, Dense
import re
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
maxlen=256
with open('reason_type_class.json','r',encoding='utf-8') as f:
    reason = json.loads(f.read())
f.close()
reason = {value: key for key, value in reason.items()}
with open('result_type_class.json', 'r', encoding='utf-8') as f:
    result = json.loads(f.read())
f.close()
result = {value: key for key, value in result.items()}
def model_dense(model_name='nezha',model_path='./best_model_dense_v1_nezha.weights',model_type='v1'):
    if model_type=='v1':
        num_classes = len(reason.keys())
    if model_type=='v2':
        num_classes = len(result.keys())
    if model_name=='nezha':
        # bert配置
        config_path = '/home/maxin/nezha_large/bert_config.json'
        checkpoint_path = '/home/maxin/nezha_large/model.ckpt'
        dict_path = '/home/maxin/nezha_large/vocab.txt'

    if model_name=='roberta':
        # bert配置
        config_path = '/home/maxin/roberta_wwm_large/bert_config.json'
        checkpoint_path = '/home/maxin/roberta_wwm_large/bert_model.ckpt'
        dict_path = '/home/maxin/roberta_wwm_large/vocab.txt'
    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    # 加载预训练模型
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False,
        model=model_name
    )

    output = Lambda(lambda x: x[:, 0])(bert.model.output)
    output = Dense(
        units=num_classes,
        activation='softmax',
        kernel_initializer=bert.initializer
    )(output)

    model = keras.models.Model(bert.model.input, output)

    model.load_weights(model_path)
    #model.summary()
    return model,tokenizer
nezha_v1_split1,nezha_v1_tokenizer = model_dense('nezha','./nezha_large_dense_v1_split1.weights','v1')
nezha_v2_split1,nezha_v2_tokenizer = model_dense('nezha','./nezha_large_dense_v2_split1.weights','v2')

def model_pred_v1(text):
    nezha_token_ids, nezha_segment_ids = nezha_v1_tokenizer.encode(text, maxlen=maxlen)
    label = nezha_v1_split1.predict([[nezha_token_ids], [nezha_segment_ids]])[0].argmax()
    #label = nezha_v1_split2.predict([[nezha_token_ids], [nezha_segment_ids]])[0].argmax()   
    return reason[label]


def model_pred_v2(text):
    nezha_token_ids, nezha_segment_ids = nezha_v2_tokenizer.encode(text, maxlen=maxlen)
    label = nezha_v2_split1.predict([[nezha_token_ids], [nezha_segment_ids]])[0].argmax()
    return result[label]


def predict_to_file(in_file, out_file):
    """输出预测结果到文件
    """
    fw = open(out_file, 'w')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            dense_result = {}
            dense_v1 = model_pred_v1(l['text'])
            dense_v2 = model_pred_v2(l['text'])
            dense_result['reason_type']=str(dense_v1)
            dense_result['result_type'] = str(dense_v2)  #'text':l['text']
            l = json.dumps({'text_id': str(l['text_id']),'text':l['text'],'result': [dense_result]},ensure_ascii=False)
            fw.write(l + '\n')
    fw.close()
if __name__ == '__main__':
    #'ccks_task2_eval_data.txt'
    predict_to_file(sys.argv[1],'dense_pred.txt')
