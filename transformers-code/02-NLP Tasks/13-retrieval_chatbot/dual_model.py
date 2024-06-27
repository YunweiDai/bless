import torch
from transformers import BertForSequenceClassification, BertPreTrainedModel, BertModel
from typing import Optional
from transformers.configuration_utils import PretrainedConfig
from torch.nn import CosineSimilarity, CosineEmbeddingLoss

# 创建自己的类，照着BertForSequenceClassification来写就行
class DualModel(BertPreTrainedModel): # 继承自BertPreTrainedModel，好像Sentence-BERT只能用他们自己设计的包

    # __init__其实只要求config，*inputs和**kwargs好像最后也没用上
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs): 
        super().__init__(config, *inputs, **kwargs)
        # 这个任务不需要分类器，所以self.classifier也没必要加，反正也就是个线性层
        self.bert = BertModel(config)
        self.post_init() # 权重初始化并允许开启梯度检查点

    def forward( # 基本上是从BertForSequenceClassification中copy过来的
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None, # 没有提供的话会自动使用内置的绝对位置编码，但这也是可学习的
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ): # Optional需要from typing import Optional才能用
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict # 这句copy过来就行

        # Step1 分别获取sentenceA和sentenceB的输入
        senA_input_ids, senB_input_ids = input_ids[:, 0], input_ids[:, 1]
        senA_attention_mask, senB_attention_mask = attention_mask[:, 0], attention_mask[:, 1]
        senA_token_type_ids, senB_token_type_ids = token_type_ids[:, 0], token_type_ids[:, 1]

        # Step2 分别获取sentenceA和sentenceB的向量表示
        senA_outputs = self.bert(
            senA_input_ids,
            attention_mask=senA_attention_mask,
            token_type_ids=senA_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 只取[CLS]位置上的输出，outputs[2:]的确是其他tokens对应的输出
        senA_pooled_output = senA_outputs[1]    # [batch_size, hidden_size]

        senB_outputs = self.bert(
            senB_input_ids,
            attention_mask=senB_attention_mask,
            token_type_ids=senB_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 只取[CLS]位置上的输出，直接使用BertForSequenceClassification的话outputs[2:]会放在logits里一起输出，所以logits一般只取logits[0]
        senB_pooled_output = senB_outputs[1]    # [batch_size, hidden_size]

        # step3 计算相似度作为logits
        cos = CosineSimilarity()(senA_pooled_output, senB_pooled_output)    # [batch, ]

        # step4 计算loss
        
        loss = None
        if labels is not None: # 如果提供了labels那才计算loss，不然就像测试集一样
            # 小于margin的负样本被认为是简单样本，不计算loss，这是为了确保负样本对之间有一定的差异，不一定完全不相关或者负相关，而正样本对就是越相似越好！
            loss_fct = CosineEmbeddingLoss(0.3) 
            loss = loss_fct(senA_pooled_output, senB_pooled_output, labels)

        output = (cos,)
        return ((loss,) + output) if loss is not None else output # 如果要使用Trainer的话，返回要么是元组，
        # 要么是正规的SequenceClassifierOutput（return_dict存在的情况下）