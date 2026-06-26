import argparse
import sys
import re
import os
import glob
import time
import numpy as np
import torch.nn as nn
import inspect
import transformers
from torch import bfloat16, float16
from datetime import datetime

from src.utils.privit import *
from src.cfg.constants import *
from src.utils.print_utils import *
from src.utils.rag_metrics import record_metric
from src.rag.runtime import get_runtime


from typing import Optional
import requests
import huggingface_hub
from huggingface_hub import InferenceClient
import textwrap
from transformers import AutoTokenizer
from google import genai
from google.genai import types

from huggingface_hub.utils import HfHubHTTPError
import os, time, random
import json

def retrieve_base_code(idx):
    """Retrieves base code for quality control."""
    base_network = SEED_NETWORK
    return split_file(base_network)[1:][idx].strip()

def clean_code_from_llm(code_from_llm):
    """Cleans the code received from LLM."""
    try:
        # Extract and clean code assuming it is enclosed in triple backticks
        return '\n'.join(code_from_llm.strip().split("```")[1].split('\n')[1:]).strip()
    except (IndexError, AttributeError) as e:
        # Print an error message if the code extraction fails
        print("Runtime Error: No code was generated or the format is incorrect.")
        return "ERROR"  # Return ERROR
        #return ""

def get_llm_code_generator(llm_model):
    # Prefer the local uvicorn-hosted model when configured
    if LOCAL_LLM:
        if llm_model in (LLM_MIXTRAL, LLM_LLAMA3, 'llama3.3'):
            llm_code_generator = submit_mixtral_local
        elif llm_model == LLM_DEEPSEEK:
            llm_code_generator = submit_deepseek_local
        else:
            print("NO LLM SPECIFIED: USING LOCAL MIXTRAL")
            llm_code_generator = submit_mixtral_local
        qc_func = llm_code_qc_hf  # uses local server for QC prompt too
    elif INFERENCE_SUBMISSION is False:
        if llm_model == LLM_MIXTRAL:
            llm_code_generator = submit_mixtral
        elif llm_model == LLM_QWEN:
            llm_code_generator = submit_qwen
        elif llm_model == LLM_GEMMA2:
            llm_code_generator = submit_gemma2
        elif llm_model == LLM_GEMMA3:
            llm_code_generator = submit_gemma3
        elif llm_model == LLM_DEEPSEEK:
            llm_code_generator = submit_deepseek
        elif llm_model == LLM_LLAMA3:
            llm_code_generator = submit_llama3_hf
        else:
            print("NO LLM SPECIFIED: USING DEEPSEEK")
            llm_code_generator = submit_deepseek
        qc_func = llm_code_qc
    else:
        if llm_model == LLM_MIXTRAL:
            llm_code_generator = submit_mixtral_hf
        elif llm_model == LLM_LLAMA3:
            llm_code_generator = submit_llama3_hf
        elif llm_model == 'gemini':
            llm_code_generator = submit_gemini_api
        elif llm_model == LLM_GEMMA2:
            llm_code_generator = submit_gemma2_hf
        else:
            print("NO LLM SPECIFIED: USING HF MIXTRAL")
            llm_code_generator = submit_mixtral_hf
        qc_func = llm_code_qc_hf
    return llm_code_generator, qc_func


def extract_note(txt):
    """Extracts note from the part if present."""
    if "# -- NOTE --" in txt:
        note_txt = txt.split('# -- NOTE --')
        return '# -- NOTE --\n' + note_txt[1].strip() + '# -- NOTE --\n'
    return ''

def split_file(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Regular expression for the pattern
    pattern = r"# --OPTION--"
    parts = re.split(pattern, content)

    return parts

def _augment_template_with_rag(template_text: str, mutation_label: str | None, query_code: str | None = None) -> str:
    """
    Inject RAG context into a template when the runtime is enabled.
    """
    runtime = get_runtime()
    if runtime is None:
        box_print("RAG Runtime: None (RAG disabled or not initialized)", print_bbox_len=80, new_line_end=False)
        return template_text

    start = time.perf_counter()
    augmented_template, mutations = runtime.enhance_template(
        template=template_text,
        mutation_type=mutation_label,
        query_code=query_code,
        gene_id=None,
    )
    duration_ms = (time.perf_counter() - start) * 1000

    num_mutations = len(mutations) if mutations else 0
    box_print(f"RAG Enhancement: Retrieved {num_mutations} mutations in {duration_ms:.1f}ms",
              print_bbox_len=80, new_line_end=False)

    if num_mutations == 0:
        box_print("RAG: No mutations found in vector store (empty or first run)",
                  print_bbox_len=80, new_line_end=False)

    record_metric(
        "rag_prompt_enhancement",
        {
            "mutation_type": mutation_label,
            "retrieval_ms": duration_ms,
            "retrieved_mutations": num_mutations,
            "prompt_tokens": len(augmented_template.split()),
        },
    )
    return augmented_template


def _prepend_rag_context_to_prompt(prompt_text: str, mutation_label: str | None) -> str:
    """
    Build an instruction prefix that references top-performing mutations.
    """
    runtime = get_runtime()
    if runtime is None or not mutation_label:
        return prompt_text
    start = time.perf_counter()
    mutations = runtime.collect_context(mutation_type=mutation_label)
    duration_ms = (time.perf_counter() - start) * 1000
    if not mutations:
        return prompt_text
    context_block = runtime.format_context(mutations)
    rag_prefix = (
        "Here are some successful mutations from prior generations. "
        "Consider how their approaches might inspire your own creative solution, but feel free to explore novel directions.\n"
        f"{context_block}\n\n"
    )
    record_metric(
        "rag_prompt_rephrase_context",
        {
            "mutation_type": mutation_label,
            "retrieval_ms": duration_ms,
            "retrieved_mutations": len(mutations),
            "prompt_tokens": len(prompt_text.split()),
        },
    )
    return f"{rag_prefix}{prompt_text}"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def llm_code_qc(code_from_llm, base_code, generate_text):
    # TODO: make parameter
    template_path = os.path.join(ROOT_DIR, 'templates/llm_quality_control.txt')
    with open(template_path, 'r') as file:
        template_txt = file.read()
    # add code to be augmented
    prompt2llm = template_txt.format(code_from_llm, base_code)
    print("="*120);print(prompt2llm);print("="*120)
    
    res = generate_text(prompt2llm) # clean txt
    code_from_llm = res[0]["generated_text"]
    code_from_llm = '\n'.join(code_from_llm.strip().split("```")[1].split('\n')[1:]).strip()
    return code_from_llm


def llm_code_qc_hf(code_from_llm, base_code, generate_text=None):
    # TODO: make parameter
    fname = np.random.choice(['llm_quality_control_p.txt', 'llm_quality_control_p.txt'])
    template_path = os.path.join(ROOT_DIR, f'templates/{fname}')
    with open(template_path, 'r') as file:
        template_txt = file.read()
    # add code to be augmented
    prompt2llm = template_txt.format(code_from_llm, base_code)
    box_print("QC PROMPT TO LLM", print_bbox_len=120, new_line_end=False)
    print(prompt2llm)
    
    code_from_llm = submit_mixtral_local(prompt2llm, max_new_tokens=1500, top_p=0.1, temperature=0.1, 
                      model_id="mistralai/Mixtral-8x7B-v0.1", return_gen=False)
    box_print("TEXT FROM LLM", print_bbox_len=60, new_line_end=False)
    print(code_from_llm)
    code_from_llm = clean_code_from_llm(code_from_llm)
    return code_from_llm

def submit_mixtral_hf(txt2mixtral, max_new_tokens=1024, top_p=0.15, temperature=0.1, 
                      model_id="mistralai/Mixtral-8x7B-Instruct-v0.1", return_gen=False):
    """
    This function submits a model prompt to mixtral through the HuggingFace Inference API

    Parameters
    ----------
    txt2mixtral : str
        Prompt that will be sent to mixtral
    max_new_tokens : int, optional
       A setting to tell the LLM the maximum number of tokens to return, by default 1024
    top_p : float, optional
        _description_, by default 0.15
    temperature : float, optional
        _description_, by default 0.1
    model_id : str, optional
       Which mixtral variant to utilize for inference, by default "mistralai/Mixtral-8x7B-Instruct-v0.1"
    return_gen : bool, optional
        _description_, by default False

    Returns
    -------
    str
        Model's output from inference
    """   
    max_new_tokens = np.random.randint(900, 1300)
    os.environ['HF_API_KEY'] = DONT_SCRAPE_ME
    huggingface_hub.login(new_session=False)
    client = InferenceClient(model=model_id)
    client.headers["x-use-cache"] = "0"

    instructions = [
            {
                "role": "user",
                "content": "Provide code in Python\n" + txt2mixtral,
            },     
    ]

    tokenizer_converter = AutoTokenizer.from_pretrained(model_id)
    prompt = tokenizer_converter.apply_chat_template(instructions, tokenize=False)
    results = [client.text_generation(prompt, max_new_tokens=max_new_tokens, 
                                      return_full_text=False, 
                                      temperature=temperature, seed=101)]
    if return_gen:
        return results[0], None
    else:
        return results[0]
    
def submit_gemma2_hf(txt2mixtral, max_new_tokens=1024, top_p=0.15, temperature=0.1, 
                      model_id="google/gemma-2-27b-it", return_gen=False):
    max_new_tokens = np.random.randint(900, 1300)
    os.environ['HF_API_KEY'] = DONT_SCRAPE_ME
    huggingface_hub.login(new_session=False)
    client = InferenceClient(model=model_id)
    client.headers["x-use-cache"] = "0"

    instructions = [

            {
                "role": "user",
                "content": "Provide code in Python\n" + txt2mixtral,
            },     
    ]

    tokenizer_converter = AutoTokenizer.from_pretrained(model_id)
    prompt = tokenizer_converter.apply_chat_template(instructions, tokenize=False)
    rate_limit = True
    while rate_limit :
        try:
            results = [client.text_generation(prompt, max_new_tokens=max_new_tokens, 
                                      return_full_text=False, 
                                      temperature=temperature, seed=101)]
        except Exception as e:
            print(e, flush=True)
            time.sleep(np.random.randint(80, 300))
        else:
            rate_limit = False
        
    if return_gen:
        return results[0], None
    else:
        return results[0]
    
    
def submit_llama3_hf(txt2llama, max_new_tokens=1024, top_p=0.15, temperature=0.1, 
                      model_id="meta-llama/Meta-Llama-3.1-70B-Instruct", return_gen=False):
    """
    This function submits a model prompt to Llama3 through the HuggingFace Inference API

    Parameters
    ----------
    txt2llama : str
        Prompt that will be sent to Llama3
    max_new_tokens : int, optional
        A setting to tell the LLM the maximum number of tokens to return, by default 1024
    top_p : float, optional
        _description_, by default 0.15
    temperature : float, optional
        _description_, by default 0.1
    model_id : str, optional
        Which Llama3 variant to utilize for inference, by default "meta-llama/Meta-Llama-3.1-70B-Instruct"
    return_gen : bool, optional
        _description_, by default False

    Returns
    -------
    str
        Model's output from inference
    """    
    max_new_tokens = np.random.randint(900, 1300)
    os.environ['HF_API_KEY'] = DONT_SCRAPE_ME
    huggingface_hub.login(new_session=False)
    client = InferenceClient(model=model_id)
    client.headers["x-use-cache"] = "0"

    instructions = [

            {
                "role": "user",
                "content": "Provide code in Python\n" + txt2llama,
            },     
    ]

    tokenizer_converter = AutoTokenizer.from_pretrained(model_id)
    prompt = tokenizer_converter.apply_chat_template(instructions, tokenize=False)
    results = [client.text_generation(prompt, max_new_tokens=max_new_tokens, 
                                      return_full_text=False, 
                                      temperature=temperature, seed=101)]
    if return_gen:
        return results[0], None
    else:
        return results[0]
    
def submit_mixtral(txt2mixtral, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                   model_id="mistralai/Mixtral-8x7B-Instruct-v0.1", return_gen=False):
    max_new_tokens = np.random.randint(800, 1000)
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=bfloat16,
        device_map='auto'
    )
    model.eval()
    print(model.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False, 
        task="text-generation",
        temperature=temperature, 
        top_p=top_p,  
        top_k=0, 
        max_new_tokens=max_new_tokens, 
        repetition_penalty=1.1,
        do_sample=True,
    )

    res = generate_text(txt2mixtral)
    output_txt = res[0]["generated_text"]
    box_print("LLM OUTPUT", print_bbox_len=60, new_line_end=False)
    print(output_txt)
    box_print(f'time to load in seconds: {round(time.time()-start_time)}', print_bbox_len=120, new_line_end=False)   
    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text
    
def submit_qwen(txt2qwen, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                   model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct", return_gen=False):
    max_new_tokens = np.random.randint(800, 1000)
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=bfloat16,
        device_map='auto'
    )
    model.eval()
    print(model.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,  # if using langchain set True
        task="text-generation",
        # we pass model parameters here too
        temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        top_p=top_p,  # select from top tokens whose probability add up to 15%
        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
        repetition_penalty=1.1,  # if output begins repeating increase
        do_sample=True,
    )

    res = generate_text(txt2qwen)
    output_txt = res[0]["generated_text"]
    box_print("LLM OUTPUT", print_bbox_len=60, new_line_end=False)
    print(output_txt)
    box_print(f'time to load in seconds: {round(time.time()-start_time)}', print_bbox_len=120, new_line_end=False)   
    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text
    
def submit_deepseek(txt2qwen, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                   model_id="deepseek-ai/deepseek-coder-33b-instruct", return_gen=False):
    max_new_tokens = 3000
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        #torch_dtype=float16,
        device_map='auto',
    )
    model.eval()
    print(model.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,  # if using langchain set True
        task="text-generation",
        # we pass model parameters here too
        temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        top_p=top_p,  # select from top tokens whose probability add up to 15%
        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
        repetition_penalty=1.1,  # if output begins repeating increase
        do_sample=True,
    )

    res = generate_text(txt2qwen)
    output_txt = res[0]["generated_text"]
    box_print("LLM OUTPUT", print_bbox_len=60, new_line_end=False)
    print(output_txt)
    box_print(f'time to load in seconds: {round(time.time()-start_time)}', print_bbox_len=120, new_line_end=False)   
    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text
    
def submit_gemma2(txt2gemma, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                   model_id="google/gemma-2-27b-it", return_gen=False):
#                   model_id="/home/hice1/jli3325/scratch/.cache/huggingface/hub/models--google--gemma-2-2b-it", return_gen=False):
    max_new_tokens = 3000
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
       # torch_dtype=float16,
        device_map='auto'
    )
    model.eval()
    print(model.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,  # if using langchain set True
        task="text-generation",
        # we pass model parameters here too
        temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        top_p=top_p,  # select from top tokens whose probability add up to 15%
        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
        repetition_penalty=1.1,  # if output begins repeating increase
        do_sample=True,
    )

    res = generate_text(txt2gemma)
    output_txt = res[0]["generated_text"]
    box_print("LLM OUTPUT", print_bbox_len=60, new_line_end=False)
    print(output_txt)
    box_print(f'time to load in seconds: {round(time.time()-start_time)}', print_bbox_len=120, new_line_end=False)   
    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text
    

def submit_gemma3(txt2gemma, max_new_tokens=764, top_p=0.15, temperature=0.1, 
                   model_id="google/gemma-3-12b-it", return_gen=False):
    max_new_tokens = np.random.randint(800, 1000)
    print(f'max_new_tokens: {max_new_tokens}')
    start_time = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        # torch_dtype=float16,
        device_map='auto'
    )
    model.eval()
    print(model.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=False,  # if using langchain set True
        task="text-generation",
        # we pass model parameters here too
        temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        top_p=top_p,  # select from top tokens whose probability add up to 15%
        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=max_new_tokens,  # max number of tokens to generate in the output
        repetition_penalty=1.1,  # if output begins repeating increase
        do_sample=True,
    )

    res = generate_text(txt2gemma)
    output_txt = res[0]["generated_text"]
    box_print("LLM OUTPUT", print_bbox_len=60, new_line_end=False)
    print(output_txt)
    box_print(f'time to load in seconds: {round(time.time()-start_time)}', print_bbox_len=120, new_line_end=False)   
    if return_gen is False:
        return output_txt
    else:
        return output_txt, generate_text
    
def get_llm_server_hostname():
    hostname = None
    hostname_file_path = HOSTNAME_DIR 
    with open(hostname_file_path, 'r') as f:
        hostname = f.readline().strip() 
    return hostname

def submit_mixtral_local(prompt, max_new_tokens=850, temperature=0.2, top_p=0.15, server_url=None, return_gen=False):
    if server_url is None:
        server_url = f"http://{os.getenv('SERVER_HOSTNAME', 'localhost')}:{PORT}/generate"
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens, # can change to random between 800 - 1000 if needed
        "temperature": temperature,
        "top_p": top_p
    }

    headers = {"Content-Type": "application/json"}

    llm_hostname = get_llm_server_hostname()
    
    print(llm_hostname)

    server_url = f"http://{llm_hostname}:{PORT}/generate"
    
    try:
        response = requests.post(server_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            output_txt = response.json().get("generated_text", "No output received.")
            print(f'{response.json().get("response_time_sec", "-1")} sec')
            if return_gen is False:
                return output_txt
            else:
                return output_txt, generate_text
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def submit_deepseek_local(prompt, max_new_tokens=850, temperature=0.2, top_p=0.15, server_url=None, return_gen=False):
    if server_url is None:
        server_url = f"http://{get_llm_server_hostname()}:8000/generate"
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens, # can change to random between 800 - 1000 if needed
        "temperature": temperature,
        "top_p": top_p
    }
    print(os.getenv("SERVER_HOSTNAME", "localhost"))

    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(server_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            output_txt = response.json().get("generated_text", "No output received.")
            print(f'{response.json().get("response_time_sec", "-1")} sec')
            if return_gen is False:
                return output_txt
            else:
                return output_txt, generate_text
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

    
def submit_gemini_api(txt2gemini, **kwargs):
    """
    This function submits a model prompt to Gemini through its API

    Parameters
    ----------
    txt2gemini : str
        Prompt that will be sent to Gemini

    Returns
    -------
    str
        Model's output from inference
    """   
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[txt2gemini],
        
    )
    return response.text


def mutate_prompt(llm_model, template, inference_submission=INFERENCE_SUBMISSION):
    path, filename = os.path.split(template)
    with open(template, 'r') as file:
        prompt_text = file.read()
    prompt_text = prompt_text.split("```")[0].strip()
    mutation_label = os.path.splitext(filename)[0]
    prompt_base = (
        "Rephrase the following prompt template text. "
        "Return ONLY the rephrased prompt text, do NOT include any code examples or code blocks. "
        "The output should be a prompt template that can be used to instruct an LLM to modify code. "
        "Preserve the placeholder {{}} where code should be inserted.\n\n"
        "Original prompt template:\n```\n{}\n```\n\n"
        "Rephrased prompt template (text only, no code):"
    ).format(prompt_text)
    prompt = _prepend_rag_context_to_prompt(prompt_base, mutation_label)
    temp = np.random.uniform(0.1, 0.4)

    llm_code_generator, qc_func = get_llm_code_generator(llm_model)
    print("Mutating Prompts with llm:", llm_model)
    output = llm_code_generator(prompt, temperature=temp).strip()
    if "```" in output:
        output = output.split("```")[0]
    output = output + "\n```python\n{}\n```"
    with open(os.path.join(path, "mutant{}.txt".format(llm_model)), 'w') as file:
        file.write(output)


def generate_augmented_code(txt2llm, augment_idx, apply_quality_control, top_p, llm_model, temperature, mutation_label=None):
    """
    Generate augmented code using the specified LLM model.

    Parameters
    ----------
    txt2llm : str
        The prompt text to send to the LLM (should already be RAG-augmented)
    augment_idx : int
        Index of the code segment being augmented (for quality control)
    apply_quality_control : bool
        Whether to apply quality control to the generated code
    top_p : float
        Top-p sampling parameter
    llm_model : str
        LLM model identifier
    temperature : float
        Temperature parameter for generation
    mutation_label : str, optional
        Mutation type label for metrics tracking

    Returns
    -------
    str or None
        The generated and cleaned code, or None if generation failed
    """
    llm_code_generator, qc_func = get_llm_code_generator(llm_model)

    box_print("PROMPT TO LLM (RAG-augmented)" if mutation_label else "PROMPT TO LLM",
              print_bbox_len=120, new_line_end=False)
    if mutation_label:
        print(f"[Mutation Type: {mutation_label}]")
    print(txt2llm)

    try:
        # Generate code using the selected LLM
        code_from_llm = llm_code_generator(txt2llm, top_p=top_p, temperature=temperature)

        box_print("TEXT FROM LLM", print_bbox_len=60, new_line_end=False)
        print(code_from_llm)

        # Clean the generated code
        code_from_llm = clean_code_from_llm(code_from_llm)

        # Check if cleaning was successful
        if code_from_llm == "ERROR" or not code_from_llm:
            print("Failed to extract valid code from LLM response")
            return None

        # Apply quality control if requested
        if apply_quality_control:
            base_code = retrieve_base_code(augment_idx)
            code_from_llm = qc_func(code_from_llm, base_code)

        # Record metrics for RAG-enabled generation
        if mutation_label:
            record_metric("llm_code_generation", {
                "mutation_type": mutation_label,
                "model": llm_model,
                "success": True
            })

        return code_from_llm

    except Exception as e:
        print(f"Error generating augmented code: {e}")
        if mutation_label:
            record_metric("llm_code_generation", {
                "mutation_type": mutation_label,
                "model": llm_model,
                "success": False,
                "error": str(e)
            })
        return None


def select_random_seed_model(gene_id, variant_dir, seed_models_dir, model_prefix="model"):
    """
    Select a random model from the seed models directory and copy it to the target location.

    Parameters
    ----------
    gene_id : str
        The gene ID for the new model
    variant_dir : str
        Directory where evolved models are stored
    seed_models_dir : str
        Directory containing seed models to select from
    model_prefix : str, optional
        Prefix for model files, by default "model"

    Returns
    -------
    tuple
        (success: bool, selected_seed_model: str or None)
    """
    import shutil
    import glob
    import numpy as np

    # Check if seed models directory exists
    if not os.path.exists(seed_models_dir):
        print(f"\t☠ Seed models directory does not exist: {seed_models_dir}")
        return False, None

    # Get all .py files in the seed models directory
    seed_models = glob.glob(os.path.join(seed_models_dir, f"{model_prefix}_*.py"))

    if not seed_models:
        print(f"\t☠ No seed models found in {seed_models_dir}")
        return False, None

    # Randomly select a seed model
    selected_seed = np.random.choice(seed_models)
    seed_basename = os.path.basename(selected_seed)

    # Create target path
    target_path = os.path.join(variant_dir, f"{model_prefix}_{gene_id}.py")

    # Ensure variant directory exists
    os.makedirs(variant_dir, exist_ok=True)

    # Copy the seed model to the target location
    try:
        shutil.copy2(selected_seed, target_path)
        print(f"\t☑ Randomly selected seed model: {seed_basename}")
        print(f"\t☑ Copied to: {target_path}")
        return True, seed_basename
    except Exception as e:
        print(f"\t☠ Error copying seed model: {e}")
        return False, None
