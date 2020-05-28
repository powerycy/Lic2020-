from transformers import BertTokenizer
import torch.nn as nn
import torch.utils.data as Data
import json
import torch
import numpy as np
from tqdm import tqdm
from REmodel import REModel_sbuject_2,REModel_sbuject_1
from transformers import AdamW,WarmupLinearSchedule
from torch.optim.lr_scheduler import LambdaLR
from bert4keras.tokenizers import Tokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
model_path = '/home/ycy/roberta_zh'
import random
random.seed(42)
maxlen = 256
batch_size = 40
best_acc = 0.0
output_dir = '/home/ycy/HBT/save_model'
def load_data(filename):
    D = []
    with open(filename,'r',encoding='utf8') as f:
        for l in f:
            l = json.loads(l)
            d = {'text': l['text'], 'spo_list': []}
            for spo in l['spo_list']:
                for k, v in spo['object'].items():
                    d['spo_list'].append(
                        (spo['subject'], spo['predicate'] + '_' + k, v)
                    )
            D.append(d)
    return D
#加载数据集
train_data = load_data('/home/ycy/HBT/data/train_data.json')
valid_data = load_data('/home/ycy/HBT/data/dev_data.json')

#读取schema
with open('/home/ycy/HBT/data/schema.json',encoding='utf8') as f:
    id2predicate, predicate2id, n = {}, {}, 0
    predicate2type = {}
    for l in f:
        l = json.loads(l)
        predicate2type[l['predicate']] = (l['subject_type'], l['object_type'])
        for k, _ in sorted(l['object_type'].items()):
            key = l['predicate'] + '_' + k
            id2predicate[n] = key
            predicate2id[key] = n
            n += 1
# tokenizer = OurTokenizer(vocab_file=BERT_PATH + "vocab.txt")
tokenizer = BertTokenizer.from_pretrained(model_path,do_lower=True)
tokenizer_k = Tokenizer(os.path.join(model_path,'vocab.txt'), do_lower_case=True)
def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1
def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    outputs = np.array([
        np.concatenate([x, [padding] *
                        (length - len(x))]) if len(x) < length else x[:length]
        for x in inputs
    ])
    return outputs
def combine_spoes(spoes):
    """合并SPO成官方格式
    """
    new_spoes = {}
    for s, p, o in spoes:
        p1, p2 = p.split('_')
        if (s, p1) in new_spoes:
            new_spoes[(s, p1)][p2] = o
        else:
            new_spoes[(s, p1)] = {p2: o}

    return [(k[0], k[1], v) for k, v in new_spoes.items()]


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer_k.tokenize(spo[0])),
            spo[1],
            tuple(
                sorted([
                    (k, tuple(tokenizer_k.tokenize(v))) for k, v in spo[2].items()
                ])
            ),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox

def extract_spoes(text):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer_k.tokenize(text, max_length=maxlen)
    # tokens_1 = tokenizer.tokenize(text,max_length=maxlen)
    mapping = tokenizer_k.rematch(text, tokens)
    token_ids, segment_ids = tokenizer_k.encode(text, max_length=maxlen)
    # token_ids_1 = tokenizer.encode_plus(text,max_length=maxlen)['input_ids']
    token_ids_1 = token_ids
    segment_ids_1 = segment_ids
    token_ids = torch.tensor([token_ids]).to(device)
    segment_ids = torch.tensor([segment_ids]).to(device)
    subject_preds = sub_model(input_ids=token_ids, token_type_ids=segment_ids,
                                            batch_size=1,
                                            sub_train=True)

    # 抽取subject
    subject_preds = subject_preds.view(1,-1,2)
    subject_preds = subject_preds.detach().cpu().numpy()
    # print(subject_preds)
    start = np.where(subject_preds[0,:,0] > 0.3)[0]
    end = np.where(subject_preds[0,:,1] > 0.3)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
    if subjects:
        spoes = []
        token_ids = np.repeat([token_ids_1], len(subjects), 0)
        segment_ids = np.repeat([segment_ids_1], len(subjects), 0)
        segment_ids = torch.tensor(segment_ids).to(device)
        token_ids = torch.tensor(token_ids).to(device)
        subjects_len = len(subjects)
        subjects = torch.tensor(subjects).to(device)
        # 传入subject，抽取object和predicate
        subject_preds,object_preds = sub_model(input_ids=token_ids, token_type_ids=segment_ids,
                                  subject_ids=subjects, batch_size=subjects_len,
                                sub_train=True, obj_train=True)
        object_preds = object_preds.detach().cpu().numpy()
        for subject, object_pred in zip(subjects, object_preds):
            start = np.where(object_pred[:,:, 0] > 0.3)
            end = np.where(object_pred[:,:,1] > 0.3)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        if len(mapping[subject[0]]) > 0 and len(mapping[subject[1]]) > 0 and len(
                                mapping[_start]) > 0 and len(mapping[_end]) > 0:
                            spoes.append(
                                ((mapping[subject[0]][0],
                                mapping[subject[1]][-1]), predicate1,
                                (mapping[_start][0], mapping[_end][-1]))
                            )
                        break
        return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1])
                for s, p, o, in spoes]
    else:
        return []
def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred_2.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:

        R = combine_spoes(extract_spoes(d['text']))
        T = combine_spoes(d['spo_list'])
        R = set([SPO(spo) for spo in R])
        T = set([SPO(spo) for spo in T])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
                       ensure_ascii=False,
                       indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall
class data_generator:
    """数据生成器
    """

    def __init__(self, data, batch_size=batch_size, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps
    def data_res(self):
        batch_token_ids, batch_segment_ids,batch_attention_mask = [], [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        indices = list(range(len(self.data)))
        # print(len(self.data))
        np.random.shuffle(indices)
        for i in indices:
            d = self.data[i]
            token = tokenizer.encode_plus(
                d['text'], max_length=maxlen
            )
            token_ids, segment_ids,attention_mask = token['input_ids'],token['token_type_ids'],token['attention_mask']
            # 整理三元组 {s: [(o, p)]}
            spoes = {}
            for s, p, o in d['spo_list']:
                s = tokenizer.encode_plus(s)['input_ids'][1:-1]
                p = predicate2id[p]
                o = tokenizer.encode_plus(o)['input_ids'][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1, p)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # subject标签
                subject_labels = np.zeros((len(token_ids), 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1
                    subject_labels[s[1], 1] = 1
                # 随机选一个subject
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                subject_ids = (start, end)
                # 对应的object标签
                object_labels = np.zeros((len(token_ids), len(predicate2id), 2))
                for o in spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1
                # 构建batch
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                batch_subject_ids.append(subject_ids)
                batch_object_labels.append(object_labels)
                batch_attention_mask.append(attention_mask)
        batch_token_ids = sequence_padding(batch_token_ids)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        batch_subject_labels = sequence_padding(
            batch_subject_labels, padding=np.zeros(2)
        )
        batch_subject_ids = np.array(batch_subject_ids)
        batch_object_labels = sequence_padding(
            batch_object_labels,
            padding=np.zeros((len(predicate2id), 2))
        )
        batch_attention_mask = sequence_padding(batch_attention_mask)
        return [
                batch_token_ids, batch_segment_ids,
                batch_subject_labels, batch_subject_ids,
                batch_object_labels,batch_attention_mask]

class Dataset(Data.Dataset):
    def __init__(self,_batch_token_ids,_batch_segment_ids,_batch_subject_labels,_batch_subject_ids,_batch_obejct_labels,_batch_attention_mask):
        self.batch_token_data_ids = _batch_token_ids
        self.batch_segment_data_ids = _batch_segment_ids
        self.batch_subject_data_labels = _batch_subject_labels
        self.batch_subject_data_ids = _batch_subject_ids
        self.batch_object_data_labels = _batch_obejct_labels
        self.batch_attention_mask = _batch_attention_mask
        self.len = len(self.batch_token_data_ids)
    def __getitem__(self, index):
        return self.batch_token_data_ids[index],self.batch_segment_data_ids[index],self.batch_subject_data_labels[index],\
               self.batch_subject_data_ids[index],self.batch_object_data_labels[index],self.batch_attention_mask[index]
    def __len__(self):
        return self.len
def collate_fn(data):
    batch_token_ids = np.array([item[0] for item in data], np.int32)
    batch_segment_ids = np.array([item[1] for item in data], np.int32)
    batch_subject_labels = np.array([item[2] for item in data], np.int32)
    batch_subject_ids = np.array([item[3] for item in data], np.int32)
    batch_object_labels = np.array([item[4] for item in data], np.int32)
    batch_attention_mask = np.array([item[5] for item in data],np.int32)
    return {
        'batch_token_ids': torch.LongTensor(batch_token_ids),  # targets_i
        'batch_segment_ids': torch.FloatTensor(batch_segment_ids),
        'batch_subject_labels': torch.FloatTensor(batch_subject_labels),
        'batch_subject_ids': torch.LongTensor(batch_subject_ids),
        'batch_object_labels': torch.LongTensor(batch_object_labels),
        'batch_attention_mask':torch.LongTensor(batch_attention_mask)
    }
# def accuracy(out, labels):
#     outputs = np.argmax(out, axis=1) # 输出最大值
#     return f1_score(labels, outputs, labels=[0, 1], average='macro') # 分别做 f1 并取平均

dg = data_generator(train_data)
dg_dev = data_generator(valid_data)
T, S1, S2, K1, K2, M1 = dg.data_res()
torch_dataset = Dataset(T, S1, S2, K1, K2 , M1)
loader_train = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    shuffle=True,  # random shuffle for training
    num_workers=32,
    collate_fn=collate_fn,  # subprocesses for loading data
)
model_name_or_path = '/home/ycy/roberta_zh'
sub_model = REModel_sbuject_2.from_pretrained(model_name_or_path,num_labels=2,output_hidden_states=True)
no_cuda = False
device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
weight_decay = 0.01
learning_rate=4e-05
adam_epsilon=1e-05
warmup_steps= 0
epochs = 20
fp16 =False
if fp16 == True:
    sub_model.half()
# sub_model.to(device)
f = open(r'res.txt','a',encoding='utf8')
param_optimizer = list(sub_model.named_parameters())  # 打印每一次 迭代元素的名字与参数
# batch_token_ids = loader_res['batch_token_ids'].cuda()
# print(batch_token_ids)
#     # hack to remove pooler, which is not used
#     # thus it produce None grad that break apex
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': weight_decay},  # n wei 层的名称, p为参数
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # 如果是 no_decay 中的元素则衰减为 0
]
#
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)  # adamw算法
# scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))
sub_model.to(device)
sub_model.train()
train_loss = 0.0
for epoch in range(epochs):
    for setp,loader_res in tqdm(iter(enumerate(loader_train))):
        # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
        #                                  t_total=train_steps)  # warmup can su
        batch_token_ids = loader_res['batch_token_ids'].to(device)
        batch_segment_ids = loader_res['batch_segment_ids'].to(device)
        batch_subject_labels = loader_res['batch_subject_labels'].long().to(device)
        batch_subject_ids = loader_res['batch_subject_ids'].to(device)
        batch_object_labels = loader_res['batch_object_labels'].to(device)
        labels_start = (batch_subject_labels[:,:,0].to(device))
        labels_end = (batch_subject_labels[:,:,1].to(device))
        batch_attention_mask = loader_res['batch_attention_mask'].long().to(device)
        batch_segment_ids = batch_segment_ids.long().to(device)
        batch_attention_mask = batch_attention_mask.long().to(device)
        sub_out,obj_out = sub_model(input_ids=batch_token_ids,token_type_ids=batch_segment_ids,attention_mask=batch_attention_mask,labels=batch_subject_labels,
                                    subject_ids = batch_subject_ids,batch_size = batch_token_ids.size()[0],
                            obj_labels = batch_object_labels,sub_train=True,obj_train=True)
        obj_loss,scores = obj_out[0:2]
        nn.utils.clip_grad_norm_(parameters=sub_model.parameters(), max_norm=1)
        obj_loss.backward()
        # print(sub_loss)
        # print(obj_loss.item())
        train_loss += obj_loss.item()
        train_loss = round(train_loss, 4)
        optimizer.step()
        # scheduler.step()
        optimizer.zero_grad()
    sub_model.eval()
    f1, precision, recall = evaluate(valid_data)
    if f1 > best_acc :
        print("Best F1", f1)
        print("Saving Model......")
        best_acc = f1
        # Save a trained model
        model_to_save = sub_model.module if hasattr(sub_model, 'module') else sub_model  # Only save the model it-self
        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)  # 仅保存学习到的参数
        f.write(str(epoch)+'\t'+str(f1)+'\t'+str(precision)+'\t'+str(recall)+'\n')
    print(f1,precision,recall)
    f.write(str(epoch)+'\t'+str(f1)+'\t'+str(precision)+'\t'+str(recall)+'\n')


