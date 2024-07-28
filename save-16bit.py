import os
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer

if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

inputs = tokenizer(
[
    alpaca_prompt.format(
        "只用中文回答问题", # instruction
        "你知道吉安公司吗？", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

#保存为hf
model.save_pretrained_merged("outputs", tokenizer, save_method = "merged_16bit",)
