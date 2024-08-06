import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import sacrebleu
from rouge_score import rouge_scorer
import numpy as np
import html
from datasets import Dataset

SRC = 'vi'
TRG = 'en'
KIND = 'official'
name_model = "Helsinki-NLP/opus-mt-vi-en"

# load dữ liệu
def load_file(file_path):
    lang = []

    with open(file_path, "r", encoding="utf-8") as file:
        content_en = file.read()
    lang += content_en.split('\n')
    lang = [html.unescape(sent) for sent in lang]
    return lang

list_train = dict()
list_train[SRC] = load_file(f"../data/train/{SRC}.{SRC}")
list_train[TRG] = load_file(f"../data/train/{TRG}.{TRG}")

list_test = dict()
list_test[SRC] = load_file(f"../data/test/{SRC}-2013.{SRC}")
list_test[TRG] = load_file(f"../data/test/{TRG}-2013.{TRG}")

list_dev = dict()
list_dev[SRC] = load_file(f"../data/dev/{SRC}-2012.{SRC}")
list_dev[TRG] = load_file(f"../data/dev/{TRG}-2012.{TRG}")


dataset_train = Dataset.from_dict({"source": list_train[SRC], "target": list_train})
dataset_test = Dataset.from_dict({"source": list_test[SRC], "target": list_test[TRG]})
dataset_dev = Dataset.from_dict({"source": list_dev[SRC], "target": list_dev[TRG]})

#tải model:
model = AutoModelForSeq2SeqLM.from_pretrained(name_model)
tokenizer = AutoTokenizer.from_pretrained(name_model)

#tạo các phương thức cần thiết:
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

def tokenize_function(examples):
    inputs = [ex for ex in examples['source']]
    targets = [ex for ex in examples['target']]
    model_inputs = tokenizer(inputs, max_length=80, truncation=True, padding="max_length") #80 -> 128

    # Sử dụng tokenizer để mã hóa câu đích
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(text_target = targets, max_length=80, truncation=True, padding="max_length") # 80 -> 128

    # Thêm nhãn vào kết quả mã hóa
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    predictions = np.array(predictions)
    predictions[predictions < 0] = 0
    predictions = predictions.tolist()
    
    
    labels = np.array(labels)
    labels[labels < 0] = 0
    labels = labels.tolist()
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

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

# chuẩn bị dữ liệu:
tokenized_train = dataset_train.map(tokenize_function, batched=True)
tokenized_test = dataset_test.map(tokenize_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# thiết lập callback

from transformers import TrainerCallback

class CustomStepCallback(TrainerCallback):
    def __init__(self, steps_interval, action_method):
        self.steps_interval = steps_interval
        self.action_method = action_method

    def on_step_end(self, args, state, control, **kwargs):
        # Thực hiện hành động sau mỗi steps_interval bước
        if state.global_step % self.steps_interval == 0:
            self.action_method(state.global_step)

# Định nghĩa phương thức hành động
def custom_action(step):
    sentence = {
        'vi': "Chúng tôi là những bác sĩ xuất sắc và anh ấy cũng vậy, do đó ca phẫu thuật chắc chắn sẽ thành công.",
        'en': "We are excellent doctors and so is he, so the surgery will definitely be successful."
    }
    pred = predict(model, sentence[SRC], tokenizer)
    print(f"Translated: {pred}")

# Khởi tạo callback với số bước và phương thức hành động
custom_callback = CustomStepCallback(steps_interval=1000, action_method=custom_action)

# thiết lập huấn luyện
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./results-{SRC}-{TRG}-base",
    evaluation_strategy="steps",
    eval_steps=250, #100 -> 250 x
    learning_rate=3e-5, #3 -> 2
    per_device_train_batch_size=16, #8 -> 16
    per_device_eval_batch_size=16, #8 -> 16
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10, # 1 -> 10 x
    predict_with_generate=True,
    generation_max_length=50,
    save_steps=500,
    logging_dir="./logs",          # Thư mục để lưu logs
    logging_steps=250,  
    fp16 = True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train, #dev -> train x
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[custom_callback]
)

if __name__ == "__main__":
    # huấn luyện và lưu lại mô hình
    trainer.train()
    torch.save(model.state_dict(), f"{SRC}-{TRG}-parameters-{KIND}.pth")