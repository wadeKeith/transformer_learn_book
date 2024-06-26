from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./Chapter4/content/KantaiBERT",
    tokenizer="./Chapter4/content/KantaiBERT"
)


output = fill_mask("Human thinking involves human <mask>.")
print(output)

print('a')