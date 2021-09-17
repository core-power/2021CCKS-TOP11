import json
import numpy as np
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from tqdm import tqdm
from keras.utils import multi_gpu_model
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
def seed_everything(seed=0):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

#SEED = 1234
#seed_everything(SEED)

def class_num(data_path):
    with open(data_path,'r',encoding='utf-8') as f:
        data=f.readlines()
    f.close()
    dic = {}
    n = 0
    result_list =[]
    for d in data:
        d=json.loads(d)
        result_list.append(d['result'][0]['result_type'])
        dic[d['result'][0]['result_type']]= 0
    for k in dic.keys():
        dic[k]=n
        n+=1
    frequency = {}
    for key in result_list:
        frequency[key] = frequency.get(key, 0) + 1

    k = open('result_type_class.json','w',encoding='utf-8')
    k.write(json.dumps(dic,ensure_ascii=False))
    return dic,n,frequency,result_list


dic,n,frequency,result_list=class_num('/home/maxin/ccks/版本一/train_pseudo_label.txt')
num_classes = n
n_gpu = 4
maxlen = 256
batch_size = 14
epochs = 10
model_name = 'nezha_large'
model_save = model_name+'_dense_v2_split3.weights'
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
if model_name=='roberta_zh_large':
    # bert配置
    config_path = '/home/maxin/roberta_zh_large/bert_config_large.json'
    checkpoint_path = '/home/maxin/roberta_zh_large/roberta_zh_large_model.ckpt'
    dict_path ='/home/maxin/roberta_zh_large/vocab.txt'



def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            #old_list = l['text'].split('###')
            #l['text'] = sorted(old_list, key=lambda x: len(x))[-1]
            text, label = l['text'], dic[l['result'][0]['result_type']]
            D.append((text, int(label)))
    return D

prior_list = []
for i in dic.keys():
    p = int(frequency.get(i))/len(result_list)
    prior_list.append(p)


# 加载数据集
train_data = load_data(
    '/home/maxin/ccks/版本三/train_pseudo_label.txt'
)
valid_data = load_data(
    '/home/maxin/ccks/版本三/dev.txt'
)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
with tf.device('/cpu:0'):
    # 加载预训练模型
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False,
        model=model_name.split('_')[0]
    )

    output = Lambda(lambda x: x[:, 0])(bert.model.output)
    output = Dense(
        units=num_classes,
        activation='softmax',
        kernel_initializer=bert.initializer
    )(output)

    model = keras.models.Model(bert.model.input, output)
    model.summary()
m_model = multi_gpu_model(model, gpus=n_gpu)
def sparse_categorical_crossentropy_with_prior(y_true, y_pred, tau=1.0):
    """带先验分布的稀疏交叉熵
    注：y_pred不用加softmax
    """
    prior = np.array(prior_list)  # 自己定义好prior，shape为[num_classes]
    log_prior = K.constant(np.log(prior + 1e-8))
    for _ in range(K.ndim(y_pred) - 1):
        log_prior = K.expand_dims(log_prior, 0)
    y_pred = y_pred + tau * log_prior
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

m_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-5),
    metrics=['sparse_categorical_accuracy'],
)


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


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = m_model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            m_model.save_weights(model_save)
            m_model.load_weights(model_save)
            model.save_weights(model_save)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


def predict_to_file(in_file, out_file):
    """输出预测结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    fw = open(out_file, 'w')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            text = l['sentence']
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            label = model.predict([[token_ids], [segment_ids]])[0].argmax()
            l = json.dumps({'id': str(l['id']), 'label': str(label)})
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':

    evaluator = Evaluator()

    m_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
    #m_model.load_weights('best_model_dense_v2.weights')
    #model.save_weights('best_model_dense_v2.weights')
