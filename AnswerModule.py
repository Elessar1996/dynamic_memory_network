import torch

from torch import nn
class AnswerModule(nn.Module):

  '''
  This Answer Module is specifically designed for bAbI dataset in which answers
  has only one word.
  '''

  def __init__(self, embedding_size, middle_size, vocab_size):

    super().__init__()
    print(f'vocab size: {vocab_size}')
    self.embedding_size = embedding_size 
    self.middle_size = middle_size 
    self.gru = nn.GRU(input_size=embedding_size, hidden_size=embedding_size, 
                      num_layers=1, batch_first=True)
    self.softmax = nn.Softmax()
    self.linear_layer_1 = nn.Linear(in_features=embedding_size, out_features=vocab_size)
    self.linear_layer_2 = nn.Linear(in_features=middle_size+1, out_features=1)

  def forward(self, m0, q):
    
    a0 = m0[:, -1, :]

    # y0 = self.softmax(self.linear_layer(a0))

    # print(f'y0 shape: {y0.shape}')

    
    concatenated_tensor = torch.cat((q, a0.unsqueeze(dim=1)), dim=1)
    a1 = self.gru(concatenated_tensor, a0.unsqueeze(dim=0))[0]

    intermediate_a1 = self.linear_layer_1(a1)
    upper_intermediate_a1 = self.linear_layer_2(torch.transpose(intermediate_a1, 1, 2))
    advanced_a1 = upper_intermediate_a1.squeeze(dim=-1)
    y1 = self.softmax(advanced_a1)
    
    return y1 