import re
import time
import glob
import numpy as np
import transformers
from torch import bfloat16
import argparse
from cfg.constants import *
from utils.print_utils import box_print
from llm_utils import mutate_prompt

    
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Augment Python Prompt Script.')

    # Add arguments
    parser.add_argument('--llm_model', type=str, default=LLM_MIXTRAL, help='LLM Model Name')
    parser.add_argument('--template', type=str, default=None, help='File Path to prompt')

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function with the parsed arguments
    mutate_prompt(llm_model=args.llm_model, template=args.template)