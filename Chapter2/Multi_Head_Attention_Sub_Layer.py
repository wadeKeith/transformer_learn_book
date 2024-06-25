from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings(action = 'ignore')


tokenizer = AutoTokenizer.from_pretrained("KennStack01/Helsinki-NLP-opus-mt-en-zh")
model = AutoModelForSeq2SeqLM.from_pretrained("KennStack01/Helsinki-NLP-opus-mt-en-zh")
# tokenizer = AutoTokenizer.from_pretrained("KennStack01/Helsinki-NLP-opus-mt-zh-en")
# model = AutoModelForSeq2SeqLM.from_pretrained("KennStack01/Helsinki-NLP-opus-mt-zh-en")
translator = pipeline("translation_en_to_zh",
                      model=model,
                      tokenizer=tokenizer)
#One line of code!
print(translator("Hello local large model", max_length=80))