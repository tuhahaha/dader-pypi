import torch
import torch.nn as nn

class BertMatcher(nn.Module):
    """This is the matcher when Feature Extractor is LMs (Bert etc.)"""
    def __init__(self, hidden_size=None, num_labels=None, dropout=0.1):
        super(BertMatcher, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.Matcher = nn.Linear(hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, x):
        x = self.dropout(x)
        out = self.Matcher(x)
        return out

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class FlexibleMatcher(nn.Module):
    """This is the matcher that can be defined by yourself"""

    def __init__(self, hidden_size=None, num_labels=None, dropout=0.1):
        super(FlexibleMatcher, self).__init__()
        # here define the neural networks of matcher

        # here is an example
        # self.dropout = nn.Dropout(p=dropout)
        # self.Matcher = nn.Linear(hidden_size, num_labels)

        pass

    def forward(self, x):
        # here define the process of matcher
        # input: Fixed length features
        # output: classification probability

        # here is an example
        # x = self.dropout(x)
        # out = self.Matcher(x)
        # return out

        pass
