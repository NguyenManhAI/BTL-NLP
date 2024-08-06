from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Literal
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getModelandTokenizer(src_lang, trg_lang, kind):
    name_model = f"NguyenManhAI/translation-{src_lang}-{trg_lang}-{kind}"

    model = AutoModelForSeq2SeqLM.from_pretrained(name_model)
    tokenizer = AutoTokenizer.from_pretrained(name_model)

    return model, tokenizer
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

    def translation(self, sentence, src_lang: Literal['en', 'vi'], trg_lang: Literal['en', 'vi'], kind: Literal['official', 'baseline']):
        if src_lang not in ['en', 'vi'] or trg_lang not in ['en', 'vi'] or kind not in ['official', 'baseline']:
            raise ValueError("Args must be assign exactly given values")
        if  src_lang == trg_lang:
            raise ValueError("Model doesn't permit both source language and target language are same")

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