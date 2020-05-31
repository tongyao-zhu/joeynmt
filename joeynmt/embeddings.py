import math
from torch import nn, Tensor
from joeynmt.helpers import freeze_params
import torch

class Embeddings(nn.Module):

    """
    Simple embeddings class
    """

    # pylint: disable=unused-argument
    def __init__(self,
                 embedding_dim: int = 64,
                 scale: bool = False,
                 vocab_size: int = 0,
                 padding_idx: int = 1,
                 freeze: bool = False,
                 from_pretrained: bool = False,
                 pretrained_path: str = "",
                 **kwargs):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super(Embeddings, self).__init__()

        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.from_pretrained = from_pretrained
        if from_pretrained:
            print("using pretrained model")
            self.weight = torch.load(pretrained_path)
            print(f"loaded embeddings size {self.weight.shape}")
            self.lut = nn.Embedding.from_pretrained(self.weight)
        else:
            self.lut = nn.Embedding(vocab_size, self.embedding_dim,
                                    padding_idx=padding_idx)
        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """
        # print(f"normally, input tensor is {x}")
        if 0 in x and self.from_pretrained:
            print(f"input tensor is {x}")
            print("Found unknown token!")
            assert (1==0)
        if 2 in x and self.from_pretrained:
            print(f"Found BOF!, input tensor is {x}")
            assert (1==0)
        if 3 in x and self.from_pretrained:
            print(f"Found EOF!, input tensor is {x}")
            assert (1==0)
        # if 1 in x and self.from_pretrained:
        #     print(f"Found padding in sentence {x}")
        if self.from_pretrained:
            print("Using pretrained")
        
            returned_value = self.lut(x)
            print(f"Normal, x is {x}, has shape {x.shape}")
     #       indices = [int(y) + 4 for y in x]
            indices = [int(y) for y in x[0]]

            real_value = (self.weight)[indices]
            print(f"real value has shape {real_value.shape}, is {real_value}")
            print(f"returned_value[0] has shape {returned_value[0].shape}, is {returned_value[0]}")
            actual_index = []
            for index in range(len(returned_value[0])):
                real = torch.all(returned_value[0][index].cpu() == self.weight.cpu(), dim=1).nonzero().item()
                actual_index.append(real)
            print(f"the returned index in weights {actual_index}")
            assert (returned_value[0].cpu()==real_value.cpu()).all()
        if self.scale:
            return self.lut(x) * math.sqrt(self.embedding_dim)
        return self.lut(x)

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__, self.embedding_dim, self.vocab_size)
