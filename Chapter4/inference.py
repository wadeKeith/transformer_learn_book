from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./Chapter4/content/KantaiBERT",
    tokenizer="./Chapter4/content/KantaiBERT",
    # device='cuda',
    device_map='mps'
)


output = fill_mask("Human thinking involves human <mask>.")
print(output)

print('a')