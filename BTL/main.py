from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torchtext.data.utils import get_tokenizer
from underthesea import word_tokenize
from typing import Literal, List
import torch
from torch import Tensor
import torch.nn as nn
import math
from torch.nn import Transformer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
prefix = "./model/"
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

class TranslationTransformer():
    def __init__(self, src: Literal['en','vi'], trg: Literal['en', 'vi']):
        self.config = {
            'MAX_LENGTH' : 64,
            'SRC_LANGUAGE' : src,
            'TGT_LANGUAGE' : trg,
            'UNK_IDX': 0,
            'PAD_IDX': 1, 
            'BOS_IDX': 2,
            'EOS_IDX': 3,
            'special_symbols' : ['<unk>', '<pad>', '<bos>', '<eos>']
        }
        self.token_transform = {
            'en': lambda text: get_tokenizer('spacy', language='en_core_web_sm')(text)[:self.config['MAX_LENGTH']],
            'vi': lambda text: word_tokenize(text, format="text").split()[:self.config['MAX_LENGTH']]
        }
        # Tạo vocab
        self.vocab_transform = {
            self.config['SRC_LANGUAGE']: torch.load(prefix+f"vocab-{self.config['SRC_LANGUAGE']}.pth"),
            self.config['TGT_LANGUAGE']: torch.load(prefix+f"vocab-{self.config['TGT_LANGUAGE']}.pth"),
        }
        self.text_transform = {}
        for ln in [self.config['SRC_LANGUAGE'], self.config['TGT_LANGUAGE']]:
            self.text_transform[ln] = self.sequential_transforms(self.token_transform[ln], #Tokenization
                                                       self.vocab_transform[ln], #Numericalization
                                                       self.tensor_transform) # Add BOS/EOS and create tensor
        self.model = torch.load(prefix + f"model_{self.config['SRC_LANGUAGE']}_{self.config['TGT_LANGUAGE']}.pth", map_location = device)

    def generate_square_subsequent_mask(self,sz):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    # sinh ra output sử dụng greedy
    def greedy_decode(self,model, src, src_mask, max_len, start_symbol):
        src = src.to(device)
        src_mask = src_mask.to(device)

        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        for i in range(max_len-1):
            memory = memory.to(device)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(device)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == self.config['EOS_IDX']:
                break
        return ys

    def sequential_transforms(self,*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    # Thêm các kí tự đầu cuối
    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([self.config['BOS_IDX']]),
                        torch.tensor(token_ids),
                        torch.tensor([self.config['EOS_IDX']])))
    def translate(self, src_sentence: str):
        self.model.eval()

        src = self.text_transform[self.config['SRC_LANGUAGE']](src_sentence).view(-1, 1)
            
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(
            self.model,  src, src_mask, max_len=num_tokens + 5, start_symbol=self.config['BOS_IDX']).flatten()
        
        tokens = self.vocab_transform[self.config['TGT_LANGUAGE']].lookup_tokens(list(tgt_tokens.cpu().numpy()))

        predict = []
        for token in tokens:
            subs_token = token.split("_")
            if len(subs_token) > 1:
                predict += subs_token
            else:
                predict.append(token)

        predict = [item for item in predict if item not in self.config['special_symbols']]

        return " ".join(predict).replace("<bos>", "").replace("<eos>", "")
    

# Hugging face
def getModelandTokenizer(src_lang, trg_lang, kind):
    name_model = f"NguyenManhAI/translation-{src_lang}-{trg_lang}-{kind}"

    model = AutoModelForSeq2SeqLM.from_pretrained(name_model)
    tokenizer = AutoTokenizer.from_pretrained(name_model)

    return model, tokenizer
def predict(model, input_sentence, tokenizer):
    # Dịch một câu hoàn chỉnh
    # Token hóa câu đầu vào
    model.to(device)
    inputs = tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True).to(device)

    # Sử dụng mô hình để dự đoán
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens = 50)

    # Giải mã đầu ra của mô hình
    translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_sentence

class TranslationBiEnVi():
    def __init__(self):
        self.translation_box = {
            "en-vi":{
                "official": getModelandTokenizer('en', 'vi', 'official'),
                "baseline": getModelandTokenizer('en', 'vi', 'baseline')
            },
            "vi-en":{
                "official": getModelandTokenizer('vi', 'en', 'official'),
                "baseline": getModelandTokenizer('vi', 'en', 'baseline')
            }
        }
        self.transformer = {
            "en-vi": TranslationTransformer('en', 'vi'),
            "vi-en": TranslationTransformer('vi', 'en')
        }

    def translation(self, sentence, src_lang: Literal['en', 'vi'], trg_lang: Literal['en', 'vi'], kind: Literal['official', 'baseline', 'transformer']):
        if src_lang not in ['en', 'vi'] or trg_lang not in ['en', 'vi'] or kind not in ['official', 'baseline', 'transformer']:
            raise ValueError("Args must be assign exactly given values")
        if  src_lang == trg_lang:
            raise ValueError("Model doesn't permit both source language and target language are same")

        if kind == 'transformer':
            return self.transformer[f"{src_lang}-{trg_lang}"].translate(sentence)

        model, tokenizer = self.translation_box[f"{src_lang}-{trg_lang}"][kind]
        return predict(model, sentence, tokenizer)
if __name__ == "__main__":
    sentence = {
        'en': "My father is the best",
        'vi': "Bố mày là nhất"
    }
    translator = TranslationBiEnVi()
    print(f"Translation EN-VI: EN: {sentence['en']} - VI: {translator.translation(sentence['en'], 'en', 'vi', 'official')}")
    print(f"Translation VI-EN: VI: {sentence['vi']} - EN: {translator.translation(sentence['vi'], 'vi', 'en', 'official')}")
    print(f"Transformer VI-EN: VI: {sentence['vi']} - EN: {translator.translation(sentence['vi'], 'vi', 'en', 'transformer')}")
    print(f"Transformer EN-VI: EN: {sentence['en']} - VI: {translator.translation(sentence['en'], 'en', 'vi', 'transformer')}")