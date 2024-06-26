#@title Step 3: Training a Tokenizer
# %%time 
# from pathlib import Path

# from tokenizers import ByteLevelBPETokenizer

# paths = [str(x) for x in Path("./Chapter4").glob("**/*.txt")]
# # Initialize a tokenizer
# tokenizer = ByteLevelBPETokenizer()

# # Customize training
# tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
#     "<s>",
#     "<pad>",
#     "</s>",
#     "<unk>",
#     "<mask>",
# ])


# #@title Step 4: Saving the files to disk
# import os
# token_dir = './Chapter4/content/KantaiBERT'
# # if not os.path.exists(token_dir):
# #   os.makedirs(token_dir)
# tokenizer.save_model('./Chapter4/content/KantaiBERT')

#@title Step 5 Loading the Trained Tokenizer Files 
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

tokenizer = ByteLevelBPETokenizer(
    "./Chapter4/content/KantaiBERT/vocab.json",
    "./Chapter4/content/KantaiBERT/merges.txt",
)
# tokenizer.encode("The Critique of Pure Reason.").tokens
# tokenizer.encode("The Critique of Pure Reason.")
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

#@title Step 7: Defining the configuration of the Model
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("./Chapter4/content/KantaiBERT", max_length=512)

#@title Step 9: Initializing a Model From Scratch
from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

#@title Exploring the Parameters
# LP=list(model.parameters())
# lp=len(LP)
# # print(lp)
# # for p in range(0,lp):
# #   print(LP[p])

# #@title Counting the parameters
# np=0
# for p in range(0,lp):#number of tensors
#   PL2=True
#   try:
#     L2=len(LP[p][0]) #check if 2D
#   except:
#     L2=1             #not 2D but 1D
#     PL2=False
#   L1=len(LP[p])      
#   L3=L1*L2
#   np+=L3             # number of parameters per tensor
#   if PL2==True:
#     print(p,L1,L2,L3)  # displaying the sizes of the parameters
#   if PL2==False:
#     print(p,L1,L3)  # displaying the sizes of the parameters

# print(np)              # total number of parameters

#@title Step 10: Building the Dataset
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./Chapter4/kant.txt",
    block_size=128,
)

#@title Step 11: Defining a Data Collator
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

#@title Step 12: Initializing the Trainer
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./Chapter4/content/KantaiBERT",
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./Chapter4/content/KantaiBERT")



# print('a')m 