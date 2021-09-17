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
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
maxlen = 256

categories = ['industry', 'product', 'region']



def ner_model(model_name='nezha_large', model_path='./nezha_cluener_globalpointer.weights'):
    if model_name == 'nezha_base':
        # bert配置
        config_path = '/home/maxin/nezha_base/bert_config.json'
        checkpoint_path = '/home/maxin/nezha_base/model.ckpt'
        dict_path = '/home/maxin/nezha_base/vocab.txt'
    if model_name == 'roberta_large':
        # bert配置
        config_path = '/home/maxin/roberta_wwm_large/bert_config.json'
        checkpoint_path = '/home/maxin/roberta_wwm_large/bert_model.ckpt'
        dict_path = '/home/maxin/roberta_wwm_large/vocab.txt'
    if model_name == 'roberta_base':
        # bert配置
        config_path = '/home/maxin/roberta_wwm_base/bert_config.json'
        checkpoint_path = '/home/maxin/roberta_wwm_base/bert_model.ckpt'
        dict_path = '/home/maxin/roberta_wwm_base/vocab.txt'
    if model_name == 'nezha_large':
        # bert配置
        config_path = '/home/maxin/nezha_large/bert_config.json'
        checkpoint_path = '/home/maxin/nezha_large/model.ckpt'
        dict_path = '/home/maxin/nezha_large/vocab.txt'
    if model_name == 'nezha_wwm_large':
        # bert配置
        config_path = '/home/maxin/nezha_wwm_large/bert_config.json'
        checkpoint_path = '/home/maxin/nezha_wwm_large/model.ckpt'
        dict_path = '/home/maxin/nezha_wwm_large/vocab.txt'

    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    bert = build_transformer_model(config_path, checkpoint_path, model=model_name.split('_')[0])
    output = GlobalPointer(len(categories), 64)(bert.output)

    ner_model = Model(bert.input, output)
    ner_model.load_weights(model_path)
    # ner_model.summary()
    return ner_model, tokenizer

ner_nezha_base_split1, nezha_tokenizer = ner_model('nezha_base', './nezha_base_ner_split1.weights')
ner_nezha_base_split2, nezha_tokenizer = ner_model('nezha_base', './nezha_base_ner_split2.weights')
ner_nezha_base_split3, nezha_tokenizer = ner_model('nezha_base', './nezha_base_ner_split3.weights')
ner_nezha_base_split4, nezha_tokenizer = ner_model('nezha_base', './nezha_base_ner_split4.weights')
def causal_model():
    config_path = '/home/maxin/nezha_base/bert_config.json'
    checkpoint_path = '/home/maxin/nezha_base/model.ckpt'
    dict_path = '/home/maxin/nezha_base/vocab.txt'
    # 加载预训练模型
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False,
        model='nezha',
    )

    output = Lambda(lambda x: x[:, 0])(bert.model.output)
    output = Dense(
        units=1,
        activation='sigmoid',
        kernel_initializer=bert.initializer
    )(output)

    model = keras.models.Model(bert.model.input, output)
    return model


class NamedEntityRecognizer(object):
    """命名实体识别器
    """

    def recognize(self, text, threshold=0.3):
        # nezha模型
        nezha_tokens = nezha_tokenizer.tokenize(text, maxlen=512)
        nezha_mapping = nezha_tokenizer.rematch(text, nezha_tokens)
        nezha_token_ids = nezha_tokenizer.tokens_to_ids(nezha_tokens)
        nezha_segment_ids = [0] * len(nezha_token_ids)
        nezha_token_ids, nezha_segment_ids = to_array([nezha_token_ids], [nezha_segment_ids])
        # roberta模型
        #roberta_tokens = roberta_tokenizer.tokenize(text, maxlen=512)
        #roberta_mapping = roberta_tokenizer.rematch(text, roberta_tokens)
        #roberta_token_ids = roberta_tokenizer.tokens_to_ids(roberta_tokens)
        #roberta_segment_ids = [0] * len(roberta_token_ids)
        #roberta_token_ids, roberta_segment_ids = to_array([roberta_token_ids], [roberta_segment_ids])

        #scores_nezha_large_split1 = ner_nezha_large_split1.predict([nezha_token_ids, nezha_segment_ids])[0]
        #scores_nezha_large_split2 = ner_nezha_large_split2.predict([nezha_token_ids, nezha_segment_ids])[0]
        scores_nezha_base_split1 = ner_nezha_base_split1.predict([nezha_token_ids, nezha_segment_ids])[0]
        scores_nezha_base_split2 = ner_nezha_base_split2.predict([nezha_token_ids, nezha_segment_ids])[0]
        scores_nezha_base_split3 = ner_nezha_base_split3.predict([nezha_token_ids, nezha_segment_ids])[0]
        scores_nezha_base_split4 = ner_nezha_base_split4.predict([nezha_token_ids, nezha_segment_ids])[0]
        scores_nezha_base = 0.25*scores_nezha_base_split1+0.25*scores_nezha_base_split2+0.25*scores_nezha_base_split3+0.25*scores_nezha_base_split4
        scores_nezha_base[:, [0, -1]] -= np.inf
        scores_nezha_base[:, :, [0, -1]] -= np.inf

        entities = []
        
        for l, start, end in zip(*np.where(scores_nezha_base > threshold)):
            entities.append(
                (nezha_mapping[start][0], nezha_mapping[end][-1], categories[l])
            )
       
        return entities


NER = NamedEntityRecognizer()
pointmodel = causal_model()
pointmodel.load_weights('./nezha_base_causal.weights')
label_list = ['reason_industry', 'result_industry', 'reason_product', 'result_product', 'reason_region','result_region']

def ner_gp(text):

    l = {}
    for start, end, label in NER.recognize(text):
        if label not in l:
            l[label] = {}
        entity = text[start:end]
        if entity not in l[label]:
            l[label][entity] = []
        l[label][entity].append([start, end])
   
    result = {}
    for k in l.keys():
        a = list(l[k].keys())
        candidate = list(set(a))
        candidate.sort(key=a.index)
        print(candidate)
        if k == 'industry':
            reason_industry = []
            result_industry = []
            for c in candidate:
                sentence = '由于行业'+c+'。'+text
                token_ids, segment_ids = nezha_tokenizer.encode(sentence, maxlen=512)
                label1 = pointmodel.predict([[token_ids], [segment_ids]])[0][0]
                print(sentence,label1)
                if label1 >=0.8:
                    reason_industry.append(c)

                sentence ='导致行业' + c  +'。'+ text
                token_ids, segment_ids = nezha_tokenizer.encode(sentence, maxlen=512)
                label2 = pointmodel.predict([[token_ids], [segment_ids]])[0][0]
                print(sentence,label2)
                if label2>= 0.8:
                    result_industry.append(c)
            result['reason_industry']=','.join(reason_industry)
            result['result_industry'] =','.join(result_industry)
        if k == 'product':
            reason_product = []
            result_product = []
            for c in candidate:
                sentence =  '由于产品'+c +'。'+ text
                token_ids, segment_ids = nezha_tokenizer.encode(sentence, maxlen=512)
                label1 = pointmodel.predict([[token_ids], [segment_ids]])[0][0] 
                print(sentence,label1)
                if label1 >=0.8:
                    reason_product.append(c)

                sentence = '导致产品'+ c +'。'+ text
                token_ids, segment_ids = nezha_tokenizer.encode(sentence, maxlen=512)
                label2 = pointmodel.predict([[token_ids], [segment_ids]])[0][0]
                print(sentence,label2)
                if label2>=0.8:
                    result_product.append(c)

            result['reason_product'] = ','.join(reason_product)
            result['result_product'] = ','.join(result_product)
        if k == 'region':
            reason_region = []
            result_region = []
            for c in candidate:
                sentence = '由于地区'+c +'。'+ text
                token_ids, segment_ids = nezha_tokenizer.encode(sentence, maxlen=512)
                label1 = pointmodel.predict([[token_ids], [segment_ids]])[0][0]
                print(sentence,label1)
                if label1>=0.8:
                    reason_region.append(c)

                sentence = '导致地区'+c +'。'+ text
                token_ids, segment_ids = nezha_tokenizer.encode(sentence, maxlen=512)
                label2 = pointmodel.predict([[token_ids], [segment_ids]])[0][0]
                print(sentence,label2)
                if label2>=0.8:
                    result_region.append(c)

            result['reason_region'] = ','.join(reason_region)
            result['result_region'] = ','.join(result_region)

            #label_list.sort(key=lambda x: x[1], reverse=True)  # 即：根据list中元组的第二个元素进行排序
        # push_list = [i[0] for i in label_list]
        # result[k] = ','.join(push_list)
    
    for s in list(set(label_list).difference(set(result.keys()))):
        result[s] = ''
    return result


def predict_to_file(in_file, out_file,test=False):
    """输出预测结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    if test==True:
        text ='电荒导致电解铝产量增速下滑支撑铝价走强'
        ner_result = ner_gp(text)
        print(ner_result)
    else:
        fw = open(out_file, 'w')
        with open(in_file) as fr:
            for l in tqdm(fr):
                l = json.loads(l)
                ner_result = ner_gp(l['text'])
                l = json.dumps({'text_id': str(l['text_id']),'text':l['text'],'result': [ner_result]},ensure_ascii=False)
                fw.write(l + '\n')
        fw.close()
if __name__ == '__main__':
    #'ccks_task2_eval_data.txt'
    predict_to_file(sys.argv[1],'ner_pred.txt',test=False)
