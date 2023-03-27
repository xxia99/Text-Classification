# Author: Xin Xia
# Date: 3/26/2023
###########################
#  Text Classifier Model  #
###########################
import torch
from torch import nn
from transformers import AutoModel


class TextMLP(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5, use_pretrained=True, vocab_size=None):
        super(TextMLP, self).__init__()
        self.num_classes = num_classes  # num_classes is default to 2, for the binary classification task
        self.use_pretrained = use_pretrained
        if self.use_pretrained:
            self.embedding = AutoModel.from_pretrained('bert-base-uncased')
        else:
            assert vocab_size, "vocab_size is invalid"
            self.embedding = nn.Embedding(vocab_size, 768)
        self.fc1 = nn.Linear(768, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        # The embedding layer, provide two choices, the pretraind one from BERT or the one from scratch
        if self.use_pretrained:
            x = self.embedding(input_ids=input_ids, attention_mask=attention_mask)[1]
        else:
            x = torch.mean(self.embedding(input_ids), dim=1)

        # The fully-connected layers
        x = nn.ReLU()(self.fc1(x))  # ReLU activation function for the first layer
        x = self.dropout(x)  # add dropout mechanism to avoid overfitting
        x = self.softmax(self.fc2(x))  # softmax activation function for the second layer, best match for classification
        return x


if __name__ == "__main__":
    # Create a TextMLP instance using a unpretrained embedding layer.
    # Vocab_size is decided by the tokenizer and its vocab, here we use 10 for example
    model = TextMLP(use_pretrained=False, vocab_size=10)
    print("\n###### TextMLP with Unpretrained Embedding Layer ######\n")
    print(model)

    # Create a TextMLP instance using a pretrained embedding layer from BERT.
    print("\n###### TextMLP with Pretrained Embedding Layer ######\n")
    model = TextMLP()
    print(model)