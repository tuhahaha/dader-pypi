import torch
import torch.nn as nn
from transformers import BertModel
from torch.autograd import Function
from transformers import BartTokenizer, BartModel
from ._utils import shift_tokens_right
import sys
sys.path.append("..")
from .. import param

class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(param.default_bert)
    def forward(self, x, mask=None,segment=None):
        outputs = self.encoder(x, attention_mask=mask,token_type_ids=segment)
        feat = outputs[1]
        return feat

class BartPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BartEncoder(nn.Module):
    def __init__(self):
        super(BartEncoder,self).__init__()
        self.encoder=BartModel.from_pretrained(param.default_bart).encoder
        self.config=self.encoder.config
        
        self.lm_head1 = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.pooler = BartPooler(self.config)
    def forward(self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):
        
        output=self.encoder(input_ids=input_ids,attention_mask=attention_mask,
                            head_mask=head_mask,inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,output_hidden_states=output_hidden_states,
                            return_dict=return_dict)

        sequence_output = output[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        
        
        logits1=self.lm_head1(output[0])
        return output,logits1,pooled_output

class FlexibleEncoder(nn.Module):
    """This is the encoder that can be defined by yourself"""

    def __init__(self):
        super(FlexibleEncoder, self).__init__()

        # here define the neural networks of encoder

        # here is an example of using Bert directly
        # self.encoder = BertModel.from_pretrained('bert-base-multilingual-cased')

        pass

    def forward(self, x, mask=None, segment=None):
        # here define the process of encoder
        # input: sequence pair 
        # output: Fixed length features

        # here is an example of using Bert directly
        # outputs = self.encoder(x, attention_mask=mask, token_type_ids=segment)
        # feat = outputs[1]
        # return feat

        pass
