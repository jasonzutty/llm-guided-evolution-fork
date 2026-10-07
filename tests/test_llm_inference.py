import sys

import re
import os
import glob
import time
import numpy as np
import transformers
from torch import bfloat16, float16
from typing import Optional
import requests
import huggingface_hub
from huggingface_hub import InferenceClient
import textwrap
from transformers import AutoTokenizer

#MODEL="Qwen/Qwen2.5-72B-Instruct"
#MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
#MODEL="google/gemma-2-2b-it"
#MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL="google/gemma-2-27b-it"

def submit_llm(txt2llm, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                   model_id=MODEL, return_gen=False):
    max_new_tokens = np.random.randint(800, 1000)
    print("LLM being used: ", model_id)
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=bfloat16,
        device_map='auto',
        token=""
    )
    model.eval()
    print(model.device)
    #tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_auth_token=""  # include this if the model requires authentication
    )

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,  # if using langchain set True
        task="text-generation",
        temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        top_p=top_p,  # select from top tokens whose probability add up to 15%
        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
        repetition_penalty=1.1,  # if output begins repeating increase
        do_sample=True,
    )

    res = generate_text(txt2llm)
    output_txt = res[0]["generated_text"]
    print(f'time to load in seconds: {round(time.time()-start_time)}')   
    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text
    

if __name__ == "__main__":
    print("STARTING LLM TEXT GENERATION")
    text2llm = "Write a creative story about a team of college students making a breakthrough in AI."
    print("Input Text: ", text2llm)
    output_txt = submit_llm(text2llm)
    print("LLM OUTPUT: \n", output_txt)
    