from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from typing import Iterable, List
import sacrebleu
from rouge_score import rouge_scorer
import numpy as np
import torch
import html
from underthesea import word_tokenize
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from timeit import default_timer as timer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pandas as pd

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 64
SRC_LANGUAGE = 'vi'
TGT_LANGUAGE = 'en'
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
sentence = {
    'en': "The patient have a good health",
    'vi': "Những bệnh nhân này hoàn toàn khỏe mạnh"
}

token_transform = {}
vocab_transform = {}

# Tạo tokenizer cho tiếng việt, anh
english_tokenizer_func = get_tokenizer('spacy', language='en_core_web_sm')
def vietnamese_tokenizer(text):
    return word_tokenize(text, format="text").split()[:MAX_LENGTH]
def english_tokenizer(text):
    return english_tokenizer_func(text)[:MAX_LENGTH]

token_transform['en'] = english_tokenizer
token_transform['vi'] = vietnamese_tokenizer

# Tạo vocab
prefix_vocab = "./model/"
vocab_transform[SRC_LANGUAGE] = torch.load(prefix_vocab+f'vocab-{SRC_LANGUAGE}.pth')
vocab_transform[TGT_LANGUAGE] = torch.load(prefix_vocab+f'vocab-{TGT_LANGUAGE}.pth')

# Mô hình
# Mô-đun trợ giúp thêm mã hóa vị trí vào việc nhúng mã thông báo để giới thiệu khái niệm về thứ tự từ.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 100):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
# Model
# chuyển token thành vector embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

# Tạo mặt nạ cho Attention
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
# sinh ra output sử dụng greedy
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# Thêm các kí tự đầu cuối
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# chuyển string thành các chỉ số
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor
# hàm dịch
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()

    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
        
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    
    tokens = vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))

    predict = []
    for token in tokens:
        subs_token = token.split("_")
        if len(subs_token) > 1:
            predict += subs_token
        else:
            predict.append(token)

    predict = [item for item in predict if item not in special_symbols]

    return " ".join(predict).replace("<bos>", "").replace("<eos>", "")
# Tính toán các metric
def compute_metrics(decoded_preds, decoded_labels):
    """
    input: 
        decoded_preds: ['A','B','C','D']
        decoded_labels: ['a','b','c','d']
    """
    # Tính BLEU
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])

    # Tính TER
    ter = sacrebleu.corpus_ter(decoded_preds, [decoded_labels])
    
    # Tính CHRF
    chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])

    # Tính ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rougeL_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(pred, label)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    return {
        "bleu": bleu.score,
        "ter": ter.score,
        "chrf": chrf.score,
        "rouge1": avg_rouge1,
        "rougeL": avg_rougeL
    }

#Load data
def load_file(file_path):
    lang = []

    with open(file_path, "r", encoding="utf-8") as file:
        content_en = file.read()
    lang += content_en.split('\n')
    lang = [html.unescape(sent) for sent in lang]
    return lang

if __name__ == "__main__":
    # load model
    transformer = torch.load(f"./model/model_{SRC_LANGUAGE}_{TGT_LANGUAGE}.pth", map_location = DEVICE)
    print(translate(transformer, sentence[SRC_LANGUAGE]))

    prefix_translation = "../data/"

    list_test_en = load_file(prefix_translation+"test/en-2013.en")
    list_test_vi = load_file(prefix_translation+"test/vi-2013.vi")

    data = {
        'en':{
            'test': list_test_en,
        },
        'vi':{
            'test': list_test_vi,
        }
    }

    start_time = timer()
    src = data[SRC_LANGUAGE]['test']
    reference = data[TGT_LANGUAGE]['test']
    candidate = [translate(transformer, sent) for sent in src]

    translation = pd.DataFrame({
        'inputs': src,
        'preds': candidate,
        'labels': reference
    })

    metrics = compute_metrics(candidate, reference)

    result = (f"BLEU: {metrics['bleu']}, TER: {metrics['ter']}, CHRF: {metrics['chrf']}, ROUGE1: {metrics['rouge1']}, ROUGEL: {metrics['rougeL']}")

    f = open('./result/metrics.txt', 'w')
    f.write(result)
    f.close()

    translation.to_csv('./result/translation.csv', index=False)
    end_time = timer()
    print(f"Save sucessfull after {end_time-start_time}")


