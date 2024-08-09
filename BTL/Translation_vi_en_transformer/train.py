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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 64
SRC_LANGUAGE = 'vi'
TGT_LANGUAGE = 'en'
#config các tham số và siêu tham số mô hình
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 2048
BATCH_SIZE = 64 # 32 -> 128
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
sentence = {
    'en': 'we all are doctors',
    'vi': "Chúng tôi là những bác sĩ"
}

#Load data
def load_file(file_path):
    lang = []

    with open(file_path, "r", encoding="utf-8") as file:
        content_en = file.read()
    lang += content_en.split('\n')
    lang = [html.unescape(sent) for sent in lang]
    return lang
prefix_translation = "../data/"

list_train_en = load_file(prefix_translation+"train/en.en")
list_train_vi = load_file(prefix_translation+"train/vi.vi")

list_test_en = load_file(prefix_translation+"test/en-2013.en")
list_test_vi = load_file(prefix_translation+"test/vi-2013.vi")

list_dev_en = load_file(prefix_translation+"dev/en-2012.en")
list_dev_vi = load_file(prefix_translation+"dev/vi-2012.vi")

data = {
    'en':{
        'train': list_train_en,
        'test': list_test_en,
        'dev': list_dev_en
    },
    'vi':{
        'train': list_train_vi,
        'test': list_test_vi,
        'dev': list_dev_vi
    }
}

# Place-holders
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

# Xây vocab và lưu lại
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:

    for data_sample in data_iter[language]['train']:
        yield token_transform[language](data_sample)

# định nghĩa các kí tự đặc biệt
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # tạo vocab cho từng ngôn ngữ
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(data, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

torch.save(vocab_transform[SRC_LANGUAGE], f"vocab-{SRC_LANGUAGE}")
torch.save(vocab_transform[TGT_LANGUAGE], f"vocab-{TGT_LANGUAGE}")
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

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


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# build model
torch.manual_seed(0)

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

#Chuẩn bị dữ liệu

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


# chuyển các mẫu dữ liệu vào dataloader
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# Tạo dataset

class MyDataset(Dataset):
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, index):
        return self.src[index], self.trg[index]
train_dataset = MyDataset(data[SRC_LANGUAGE]['train'], data[TGT_LANGUAGE]['train'])
test_dataset = MyDataset(data[SRC_LANGUAGE]['test'], data[TGT_LANGUAGE]['test'])

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

def unsqueeze(lst: list, dim = -1):
    lst = np.expand_dims(lst, axis=dim)
    return lst.tolist()

def convert_ids_to_sentences(ids: torch.Tensor, language):
    """
    input:
        [[    2,     2,     2,  ...,     2,     2,     2],
        [ 6693,   133,    12,  ...,    48,   133,   796],
        [17755,  2869,   177,  ...,    23,   861,     6],
        ...,
        [    1,     1,     1,  ...,     1,     1,     1],
        [    1,     1,     1,  ...,     1,     1,     1],
        [    1,     1,     1,  ...,     1,     1,     1]]
    """
    sentences = []
    for ids_token in ids.transpose(0,1).tolist():
        sent = []
        for token in vocab_transform[language].lookup_tokens(ids_token):
            subs_token = token.split("_")
            if len(subs_token) > 1:
                sent += subs_token
            else:
                sent.append(token)

        sent = [item for item in sent if item not in special_symbols]
        
        sentences.append(" ".join(sent).replace("<bos>", "").replace("<eos>", "").replace("<pad>",""))

    return sentences

#tạo train và đánh giá cho từng epoch
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return f"{elapsed_mins}p{elapsed_secs}s"

def evaluate(model):
    model.eval()
    losses = 0

    val_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))

def train_epoch(model, optimizer, log_step = 100):
    losses = []
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    total = len(train_dataloader)
    step = 0
    
    current_time = timer()
    for src, tgt in train_dataloader:
        model.train()
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses.append(loss.item())
        step += 1
        if step % log_step == 0:
            val_loss = evaluate(model)
            end_time = timer()
            
            print(f"Step: {(step*100 / total):.3f}%, Train loss: {np.mean(losses):.3f}, Val loss: {val_loss:.3f}, time: {epoch_time(current_time, end_time)}")
            current_time = timer()

    return np.mean(losses)

#bắt đầu train
NUM_EPOCHS = 30 # 1 -> 18

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer) # mở lại
    val_loss = evaluate(transformer)
    print(translate(transformer, sentence[SRC_LANGUAGE]))
    end_time = timer()
    result = (f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time: {(end_time - start_time):.3f}s")
    print(result)

#Lưu mô hình
torch.save(transformer, f'model-{SRC_LANGUAGE}-{TGT_LANGUAGE}.pth')