import json
import numpy as np
from bert4keras.backend import keras, search_layer, K
#from bert4keras.backend import multilabel_categorical_crossentropy
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
from tqdm import tqdm
import re
import os
from keras.utils import multi_gpu_model
from circle_loss import SparseCircleLoss, CircleLoss, PairCircleLoss
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
#os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
def seed_everything(seed=0):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

SEED =1234  #1234
seed_everything(SEED)
train=True
n_gpu = 3
maxlen = 256
epochs = 8
batch_size = 10 #46
bert_layers = 24
learning_rate = 1e-5  # bert_layers越小，学习率应该要越大
model_name = 'nezha_large'
model_save = './'+model_name+'_ner_split1.weights'
categories = set()

if model_name=='nezha_base':
    # bert配置
    config_path = '/home/maxin/nezha_base/bert_config.json'
    checkpoint_path = '/home/maxin/nezha_base/model.ckpt'
    dict_path ='/home/maxin/nezha_base/vocab.txt'
if model_name == 'roberta_large':
    # bert配置
    config_path = '/home/maxin/roberta_wwm_large/bert_config.json'
    checkpoint_path = '/home/maxin/roberta_wwm_large/bert_model.ckpt'
    dict_path ='/home/maxin/roberta_wwm_large/vocab.txt'
if model_name == 'roberta_base':
    # bert配置
    config_path = '/home/maxin/roberta_wwm_base/bert_config.json'
    checkpoint_path = '/home/maxin/roberta_wwm_base/bert_model.ckpt'
    dict_path ='/home/maxin/roberta_wwm_base/vocab.txt'
if model_name=='nezha_large':
    # bert配置
    config_path = '/home/maxin/nezha_large/bert_config.json'
    checkpoint_path = '/home/maxin/nezha_large/model.ckpt'
    dict_path ='/home/maxin/nezha_large/vocab.txt'
if model_name=='roberta_zh_large':
    # bert配置
    config_path = '/home/maxin/roberta_zh_large/bert_config_large.json'
    checkpoint_path = '/home/maxin/roberta_zh_large/roberta_zh_large_model.ckpt'
    dict_path ='/home/maxin/roberta_zh_large/vocab.txt'
if model_name=='nezha_wwm_large':
    # bert配置
    config_path = '/home/maxin/nezha_wwm_large/bert_config.json'
    checkpoint_path = '/home/maxin/nezha_wwm_large/model.ckpt'
    dict_path ='/home/maxin/nezha_wwm_large/vocab.txt'
if model_name=='nezha_wwm_base':
    # bert配置
    config_path = '/home/maxin/nezha_wwm_base/bert_config.json'
    checkpoint_path = '/home/maxin/nezha_wwm_base/model.ckpt'
    dict_path ='/home/maxin/nezha_wwm_base/vocab.txt'



def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            s=[]
            l =json.loads(l)
            #old_list = l['text'].split('###')
            #l['text'] = sorted(old_list, key=lambda x: len(x))[-1]
            s.append(l['text'])
            if not l:
                continue
            for i in l['result']:
                categories.add('product')
                categories.add('region')
                categories.add('industry')
                if i['reason_product']!='':
                    for j in i['reason_product'].split(','):
                        for match in re.finditer(j, l['text']):
                            if match!=None:
                                s.append((match.start(), match.end(),'product'))
                            else:pass
                if i['result_product']!='':
                    for j in i['result_product'].split(','):
                        for match in re.finditer(j, l['text']):
                            if match!=None:
                                s.append((match.start(), match.end(),'product'))
                            else:pass
                if i['reason_region']!='':
                    for j in i['reason_region'].split(','):
                        for match in re.finditer(j, l['text']):
                            if match!=None:
                                s.append((match.start(), match.end(),'region'))
                            else:pass
                if i['result_region']!='':
                    for j in i['result_region'].split(','):
                        for match in re.finditer(j, l['text']):
                            if match!=None:
                                s.append((match.start(), match.end(),'region'))
                            else:pass
                if i['reason_industry']!='':
                    for j in i['reason_industry'].split(','):
                        for match in re.finditer(j, l['text']):
                            if match!=None:
                                s.append((match.start(), match.end(),'industry'))
                            else:pass
                if i['result_industry']!='':
                    for j in i['result_industry'].split(','):
                        for match in re.finditer(j, l['text']):
                            if match!=None:
                                s.append((match.start(), match.end(),'industry'))
                            else:pass
            D.append(s)
    return D
#data = load_data('../ccks/ccks_task2_train.txt')
train_data= load_data('../ccks/版本一/ner_train.txt')
valid_data = load_data('../ccks/版本一/dev.txt')
categories = list(sorted(categories))
print(categories)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros((len(categories), maxlen, maxlen))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = categories.index(label)
                    labels[label, start, end] = 1
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, seq_dims=3)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    neg_loss = K.logsumexp(y_pred_neg, axis=-1)
    pos_loss = K.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss

def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)

with tf.device('/cpu:0'):
    model = build_transformer_model(config_path, checkpoint_path,model=model_name.split('_')[0])
    output = GlobalPointer(len(categories), 64)(model.output)

    model = Model(model.input, output)
    model.summary()
m_model = multi_gpu_model(model, gpus=n_gpu)
opt = Adam(learning_rate)
#opt=tf.compat.v1.train.AdamOptimizer(learning_rate)
# add a line
#opt = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(opt,loss_scale='dynamic')
m_model.compile(
    loss=global_pointer_crossentropy,
    optimizer=opt,
    metrics=[global_pointer_f1_score]
)


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, text, threshold=0):
        tokens = tokenizer.tokenize(text, maxlen=512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        scores = m_model.predict([token_ids, segment_ids])[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*np.where(scores > threshold)):
            entities.append(
                (mapping[start][0], mapping[end][-1], categories[l])
            )
        return entities


NER = NamedEntityRecognizer()


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(NER.recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            m_model.save_weights(model_save)
            m_model.load_weights(model_save)
            model.save_weights(model_save)
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


def predict_to_file(in_file, out_file):
    """预测到文件
    可以提交到 https://www.cluebenchmarks.com/ner.html
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            l['label'] = {}
            for start, end, label in NER.recognize(l['text']):
                if label not in l['label']:
                    l['label'][label] = {}
                entity = l['text'][start:end + 1]
                if entity not in l['label'][label]:
                    l['label'][label][entity] = []
                l['label'][label][entity].append([start, end])
            l = json.dumps(l, ensure_ascii=False)
            fw.write(l + '\n')
    fw.close()

def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


# 写好函数后，启用对抗训练只需要一行代码
adversarial_training(m_model, 'Embedding-Token', 1)
if __name__ == '__main__':
    if train==True:
        evaluator = Evaluator()
        train_generator = data_generator(train_data, batch_size)

        m_model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[evaluator]
        )
        #m_model.load_weights('./best_model_cluener_globalpointer.weights')
        #model.save_weights('./best_model_cluener_globalpointer.weights')

    else:
        model.load_weights(model_save)
        print(NER.recognize('相反，上海工业部门的结构性调整导致市场对于原水需求量的下降，因此，原水价格在近几年呈下降趋势'))

