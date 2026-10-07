# This file is for reusable logic that parses Slurm logs 
# and similar text-based run logs. It should contain 
# functions to extract job IDs, classify failures such 
# as OOM or traceback errors, detect host or partition information, 
# and capture relevant first or last lines from a log. 
# Its purpose is to centralize all log-specific parsing so 
# multiple scripts can use the same consistent rules.