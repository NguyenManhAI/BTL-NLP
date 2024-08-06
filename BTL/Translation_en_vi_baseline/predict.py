import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import sacrebleu
from rouge_score import rouge_scorer
import numpy as np
import html
from datasets import Dataset
import pandas as pd

SRC = 'en'
TRG = 'vi'
KIND = 'baseline'

def load_file(file_path):
    lang = []

    with open(file_path, "r", encoding="utf-8") as file:
        content_en = file.read()
    lang += content_en.split('\n')
    lang = [html.unescape(sent) for sent in lang]
    return lang

#tải model:
name_model = f"NguyenManhAI/translation-{SRC}-{TRG}-{KIND}"
model = AutoModelForSeq2SeqLM.from_pretrained(name_model)
tokenizer = AutoTokenizer.from_pretrained(name_model)

list_test = dict()
list_test[SRC] = load_file(f"../data/test/{SRC}-2013.{SRC}")
list_test[TRG] = load_file(f"../data/test/{TRG}-2013.{TRG}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def predict(model, input_sentence, tokenizer):
    # Dịch một câu hoàn chỉnh
    # Token hóa câu đầu vào
    inputs = tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True).to(device)

    # Sử dụng mô hình để dự đoán
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens = 50)

    # Giải mã đầu ra của mô hình
    translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_sentence

if __name__ == "__main__":
    sentence = {
        'en': "I'm a doctor and so he is",
        'vi': "Tôi là một bác sĩ và anh ấy cũng vậy."
    }
    
    pred = predict(model, sentence[SRC], tokenizer)
    print(pred)

    translation = {
        'inputs':[],
        'preds':[],
        'labels':[]
    }

    for i in range(len(list_test[SRC])):
        translation['inputs'].append(list_test[SRC][i])
        translation['preds'].append(predict(model, list_test[SRC][i], tokenizer))
        translation['labels'].append(list_test[TRG][i])

    # Tính BLEU
    bleu = sacrebleu.corpus_bleu(translation['preds'], [translation['labels']])
    # Tính TER
    ter = sacrebleu.corpus_ter(translation['preds'], [translation['labels']])
    # Tính CHRF
    chrf = sacrebleu.corpus_chrf(translation['preds'], [translation['labels']])

    # Tính ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rougeL_scores = []
    for pred, label in zip(translation['preds'], translation['labels']):
        scores = scorer.score(pred, label)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    
    metric_str = f"bleu\tter\tchrf\trouge1\trougeL\n{bleu.score}\t{ter.score}\t{chrf.score}\t{avg_rouge1}\t{avg_rougeL}"

    f = open('final-result\metric.txt', 'w', encoding='utf-8')
    f.write(metric_str)
    f.close()

    pd.DataFrame(translation).to_csv('final-result/translation.csv', index=False)

    print("Lưu thành công")