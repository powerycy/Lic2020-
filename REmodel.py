from transformers import BertModel,BertPreTrainedModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import BCELoss
import random
random.seed(42)
device = torch.device('cuda')
BertLayerNorm = torch.nn.LayerNorm
def merge_function(inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
        output += inputs[i]
    return output / len(inputs)
def batch_gather(data:torch.Tensor,index:torch.Tensor):
    index = index.unsqueeze(-1)
    index = index.expand(data.size()[0],index.size()[1],data.size()[2])
    return torch.gather(data,1,index)
def extrac_subject(output, subject_ids):
    """根据subject_ids从output中取出subject的向量表征
    """
    start = batch_gather(output, subject_ids[:, :1])
    end = batch_gather(output, subject_ids[:, 1:])
    so_res = merge_function([start,end])
    # subject = torch.cat([start, end], 2)
    # return start,end
    return so_res

def extrac_subject_1(output, subject_ids):
    """根据subject_ids从output中取出subject的向量表征
    """
    start = batch_gather(output, subject_ids[:, :1])
    end = batch_gather(output, subject_ids[:, 1:])
    # so_res = merge_function([start,end])
    # subject = torch.cat([start, end], 2)
    return start,end
    # return so_res

class REModel_sbuject(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.obj_labels = 110
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.classifier = nn.Linear(config.hidden_size,config.num_labels)
        self.obj_classifier = nn.Linear(config.hidden_size, self.obj_labels)
        self.sub_pos_emb = nn.Embedding(256, 768)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(768, 768)
        self.init_weights()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        subject_ids = None,
        batch_size = None,
        obj_labels = None,
        sub_train = False,
        obj_train = False
    ):

        outputs_1 = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs_1[0]
        sequence_output = self.dropout(sequence_output)
        if sub_train == True:
            logits = self.classifier(sequence_output)
            outputs = (logits,)   # add hidden states and attention if they are here
            loss_fct = BCELoss(reduction='none')
            # loss_fct = BCEWithLogitsLoss(reduction='none')
            loss_sig = nn.Sigmoid()
            # Only keep active parts of the loss
            active_logits = logits.view(-1, self.num_labels)
            active_logits = loss_sig(active_logits)
            active_logits = active_logits ** 2
            if labels is not None :
                active_labels = labels.view(-1, self.num_labels).float()
                loss = loss_fct(active_logits, active_labels)
                # loss = loss.view(-1,sequence_output.size()[1],2)
                loss = loss.view(batch_size, -1, 2)
                loss = torch.mean(loss, 2)
                loss = torch.sum(attention_mask * loss) / torch.sum(attention_mask)
                outputs = (loss,) + outputs
            else:
                outputs = active_logits
        if obj_train == True:
            hidden_states = outputs_1[2][-2]
            # hidden_states = self.dropout(hidden_states)
            loss_obj = BCELoss(reduction='none')
            loss_sig = nn.Sigmoid()
            # sub_pos_start = self.sub_pos_emb(subject_ids[:, :1]).to(device)
            # sub_pos_end = self.sub_pos_emb(subject_ids[:, 1:]).to(device)
            # subject_start,subject_end = extrac_subject_1(hidden_states, subject_ids)
            # subject = extrac_subject(hidden_states, subject_ids)
            # subject_start = subject_start.to(device)
            # subject_end = subject_end.to(device)
            # subject = (sub_pos_start + subject_start + sub_pos_end + subject_end).to(device)
            subject = extrac_subject(hidden_states, subject_ids).to(device)
            batch_token_ids_obj = torch.add(hidden_states, subject)
            # batch_token_ids_obj = self.LayerNorm(batch_token_ids_obj)
            # batch_token_ids_obj = self.dropout(batch_token_ids_obj)
            # batch_token_ids_obj = F.dropout(batch_token_ids_obj,p=0.5)
            # batch_token_ids_obj = self.relu(self.linear(batch_token_ids_obj))
            # batch_token_ids_obj = self.dropout(batch_token_ids_obj)
            obj_logits = self.obj_classifier(batch_token_ids_obj)
            # obj_logits = self.dropout(obj_logits)
            # obj_logits = F.dropout(obj_logits,p=0.4)
            obj_logits = loss_sig(obj_logits)
            obj_logits = obj_logits ** 4
            obj_outputs = (obj_logits,)
            if obj_labels is not None:
                # obj_loss = loss_obj(obj_logits.view(-1, hidden_states.size()[1], self.obj_labels // 2, 2), obj_labels.float())
                obj_loss = loss_obj(obj_logits.view(batch_size, -1, self.obj_labels // 2, 2), obj_labels.float())
                obj_loss = torch.sum(torch.mean(obj_loss, 3), 2)
                obj_loss = torch.sum(obj_loss * attention_mask) / torch.sum(attention_mask)
                s_o_loss = torch.add(obj_loss, loss)
                outputs_obj = (s_o_loss,) + obj_outputs
            else:
                # outputs_obj = obj_logits.view(-1,hidden_states.size()[1],self.obj_labels // 2 ,2)
                outputs_obj = obj_logits.view(batch_size, -1, self.obj_labels // 2, 2)
        if obj_train == True:
            return outputs ,outputs_obj # (loss), scores, (hidden_states), (attentions)
        else:
            return outputs
class REModel_sbuject_1(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.obj_labels = 110
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.classifier = nn.Linear(config.hidden_size,config.num_labels)
        self.obj_classifier = nn.Linear(config.hidden_size, self.obj_labels)
        self.sub_pos_emb = nn.Embedding(256, 768)
        self.init_weights()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        subject_ids = None,
        batch_size = None,
        obj_labels = None,
        sub_train = False,
        obj_train = False
    ):

        outputs_1 = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs_1[0]
        sequence_output = self.dropout(sequence_output)
        if sub_train == True:
            logits = self.classifier(sequence_output)
            outputs = (logits,)   # add hidden states and attention if they are here
            loss_fct = BCELoss(reduction='none')
            # loss_fct = BCEWithLogitsLoss(reduction='none')
            loss_sig = nn.Sigmoid()
            # Only keep active parts of the loss
            active_logits = logits.view(-1, self.num_labels)
            active_logits = loss_sig(active_logits)
            active_logits = active_logits ** 2
            if labels is not None :
                active_labels = labels.view(-1, self.num_labels).float()
                loss = loss_fct(active_logits, active_labels)
                loss = loss.view(batch_size,-1,2)
                loss = torch.mean(loss, 2)
                loss = torch.sum(attention_mask * loss) / torch.sum(attention_mask)
                outputs = (loss,) + outputs
            else:
                outputs = active_logits
        if obj_train == True:
            hidden_states = outputs_1[2][-2]
            loss_obj = BCELoss(reduction='none')
            sub_pos_start = self.sub_pos_emb(subject_ids[:, :1])
            sub_pos_end = self.sub_pos_emb(subject_ids[:, 1:])
            sub_pos = sub_pos_start + sub_pos_end
            # sub_pos = sub_pos.expand(batch_size,)
            loss_sig = nn.Sigmoid()
            subject = extrac_subject(hidden_states, subject_ids)
            # subject = subject_start + subject_end
            # subject = torch.add(subject,sub_pos)
            batch_token_ids_obj = torch.add(hidden_states, subject)
            batch_token_ids_obj = self.LayerNorm(batch_token_ids_obj)
            batch_token_ids_obj = self.dropout(batch_token_ids_obj)
            obj_logits = self.obj_classifier(batch_token_ids_obj)
            obj_logits = F.dropout(obj_logits,p=0.4)
            obj_logits = loss_sig(obj_logits)
            obj_logits = obj_logits ** 4
            obj_outputs = (obj_logits,)
            if obj_labels is not None:
                obj_loss = loss_obj(obj_logits.view(batch_size, -1, self.obj_labels // 2, 2), obj_labels.float())
                obj_loss = torch.sum(torch.mean(obj_loss, 3), 2)
                obj_loss = torch.sum(obj_loss * attention_mask) / torch.sum(attention_mask)
                s_o_loss = torch.add(obj_loss, loss)
                outputs_obj = (s_o_loss,) + obj_outputs
            else:
                outputs_obj = obj_logits.view(batch_size,-1,self.obj_labels // 2 ,2)
        if obj_train == True:
            return outputs ,outputs_obj # (loss), scores, (hidden_states), (attentions)
        else:
            return outputs
class REModel_sbuject_2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.obj_labels = 110
        self.bert = BertModel(config)
        self.linear = nn.Linear(768, 768)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.classifier = nn.Linear(config.hidden_size,config.num_labels)
        self.obj_classifier = nn.Linear(config.hidden_size, self.obj_labels)
        self.sub_pos_emb = nn.Embedding(256, 768)
        self.relu = nn.ReLU()
        self.init_weights()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        subject_ids = None,
        batch_size = None,
        obj_labels = None,
        sub_train = False,
        obj_train = False
    ):

        outputs_1 = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs_1[0]
        sequence_output = self.dropout(sequence_output)
        if sub_train == True:
            logits = self.classifier(sequence_output)
            outputs = (logits,)   # add hidden states and attention if they are here
            loss_fct = BCELoss(reduction='none')
            # loss_fct = BCEWithLogitsLoss(reduction='none')
            loss_sig = nn.Sigmoid()
            # Only keep active parts of the loss
            active_logits = logits.view(-1, self.num_labels)
            active_logits = loss_sig(active_logits)
            active_logits = active_logits ** 2
            if labels is not None :
                active_labels = labels.view(-1, self.num_labels).float()
                loss = loss_fct(active_logits, active_labels)
                # loss = loss.view(-1,sequence_output.size()[1],2)
                loss = loss.view(batch_size, -1, 2)
                loss = torch.mean(loss, 2)
                loss = torch.sum(attention_mask * loss) / torch.sum(attention_mask)
                outputs = (loss,) + outputs
            else:
                outputs = active_logits
        if obj_train == True:
            hidden_states = outputs_1[2][-2]
            hidden_states_1 = outputs_1[2][-3]
            # hidden_states = self.dropout(hidden_states)
            loss_obj = BCELoss(reduction='none')
            loss_sig = nn.Sigmoid()
            sub_pos_start = self.sub_pos_emb(subject_ids[:, :1]).to(device)
            sub_pos_end = self.sub_pos_emb(subject_ids[:, 1:]).to(device)
            subject_start_last, subject_end_last = extrac_subject_1(sequence_output, subject_ids)
            subject_start_1,subject_end_1 = extrac_subject_1(hidden_states_1, subject_ids)
            subject_start,subject_end = extrac_subject_1(hidden_states, subject_ids)
            # subject = extrac_subject(hidden_states, subject_ids)
            # subject_start = subject_start.to(device)
            # subject_end = subject_end.to(device)
            subject = (sub_pos_start + subject_start + sub_pos_end + subject_end + subject_start_last + subject_start_1 + subject_end_1 + subject_end_1).to(device)
            # subject = extrac_subject(sequence_output, subject_ids).to(device)
            batch_token_ids_obj = torch.add(hidden_states, subject)
            batch_token_ids_obj = self.LayerNorm(batch_token_ids_obj)
            batch_token_ids_obj = self.dropout(batch_token_ids_obj)
            # batch_token_ids_obj = F.dropout(batch_token_ids_obj,p=0.5)
            batch_token_ids_obj = self.relu(self.linear(batch_token_ids_obj))
            batch_token_ids_obj = self.dropout(batch_token_ids_obj)
            # batch_token_ids_obj = F.dropout(batch_token_ids_obj,p=0.4)
            obj_logits = self.obj_classifier(batch_token_ids_obj)

            obj_logits = loss_sig(obj_logits)
            obj_logits = obj_logits ** 4
            obj_outputs = (obj_logits,)
            if obj_labels is not None:
                # obj_loss = loss_obj(obj_logits.view(-1, hidden_states.size()[1], self.obj_labels // 2, 2), obj_labels.float())
                obj_loss = loss_obj(obj_logits.view(batch_size, -1, self.obj_labels // 2, 2), obj_labels.float())
                obj_loss = torch.sum(torch.mean(obj_loss, 3), 2)
                obj_loss = torch.sum(obj_loss * attention_mask) / torch.sum(attention_mask)
                s_o_loss = torch.add(obj_loss, loss)
                outputs_obj = (s_o_loss,) + obj_outputs
            else:
                # outputs_obj = obj_logits.view(-1,hidden_states.size()[1],self.obj_labels // 2 ,2)
                outputs_obj = obj_logits.view(batch_size, -1, self.obj_labels // 2, 2)
        if obj_train == True:
            return outputs ,outputs_obj # (loss), scores, (hidden_states), (attentions)
        else:
            return outputs