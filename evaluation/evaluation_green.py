#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import argparse
import pandas as pd
from tqdm import tqdm
from green_score import GREEN
from transformers import AutoTokenizer



# -----------------------------------------------------------------------------
# 2) Command-line arguments
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="GREEN score of the model outputs against the reference answers.")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV file with the columns [Ground Truth, <model_name>].")
    parser.add_argument("--output_dir", type=str, required=True, help="Output CSV file with one GREEN score per row.")
    parser.add_argument("--model_name", type=str, required=True, help="Column of the input CSV that holds the model outputs.")
    return parser.parse_args()

def truncate_text(text, tokenizer, max_length):
    if not isinstance(text, str):
        text = str(text) if text is not None else "nan"
    if text.strip() == "":
        return "nan"
    
    inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    if len(inputs.input_ids[0]) > max_length:
        print("check == ", len(inputs.input_ids[0]))
    
    # decode with batch_decode
    tokenized_text = tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True)[0]
    return tokenized_text.strip()


# -----------------------------------------------------------------------------
# 3) Main
# -----------------------------------------------------------------------------
def main():
    # (1) Parse the arguments
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    model_name = "StanfordAIMI/GREEN-radllama2-7b"

    # load the tokenizer
    tokenizer  = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # tokenizer  = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
    # tokenizer.padding_side = "left"
    max_length = tokenizer.model_max_length - 768  # usually 2048
    print("max_length == ", max_length)
    refs = df['Ground Truth'].astype(str).tolist()
    hyps = df[args.model_name].astype(str).tolist()

    # truncate the references and the model outputs
    refs = [truncate_text(ref, tokenizer, max_length).replace("<s>", "").strip() for ref in refs]
    hyps = [truncate_text(hyp, tokenizer, max_length).replace("<s>", "").strip() for hyp in hyps]

    green_scorer = GREEN(model_name, output_dir=".")

    _, _, green_score_list, _, _ = green_scorer(refs, hyps)

    eval_df = pd.DataFrame()
    eval_df['result'] = green_score_list
    
    eval_df.to_csv(args.output_dir, index=False)


# -----------------------------------------------------------------------------
# 4) Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
