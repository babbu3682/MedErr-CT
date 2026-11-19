#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
import dspy


# -----------------------------------------------------------------------------
# 1) Evaluation signature
# -----------------------------------------------------------------------------
class Evaluation(dspy.Signature):
    """
    You are given a ground truth medical report and a predicted medical report. 
    Your job is to evaluate the predicted report.
    
    Identify the aspects mentioned in the ground truth, and calculate the 
    percentage of these aspects that are either correctly mentioned or 
    partially matched in the prediction, scoring from 0 to 100.
    """
    ground_truth_report: str = dspy.InputField(desc="Ground truth medical report")
    predicted_report: str = dspy.InputField(desc="Predicted medical report")
    
    score: int = dspy.OutputField(desc="Score from 0 to 100")


# -----------------------------------------------------------------------------
# 2) Command-line arguments
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Clinical-accuracy score of the model outputs with GPT-4o mini via DSPy.")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV file with the columns [Ground Truth, <target_model>].")
    parser.add_argument("--output_dir", type=str, required=True, help="Output CSV file with one score (0-100) per row.")
    parser.add_argument("--target_model", type=str, required=True, help="Column of the input CSV that holds the model outputs.")
    
    return parser.parse_args()


# -----------------------------------------------------------------------------
# 3) Main
# -----------------------------------------------------------------------------
def main():
    # (1) Parse the arguments
    args = parse_args()

    # (2) OpenAI API key
    assert os.environ.get("OPENAI_API_KEY"), "set the environment variable OPENAI_API_KEY"

    # (3) DSPy model
    lm = dspy.LM("openai/gpt-4o-mini", model_type="chat")
    dspy.configure(lm=lm)
    evaluator = dspy.ChainOfThought(Evaluation)

    # (4) Load the CSV file
    df_full_path = args.input_csv
    if not os.path.exists(df_full_path):
        raise FileNotFoundError(f"Input CSV file not found: {df_full_path}")
    df_full = pd.read_csv(df_full_path)

    # (5) Score each pair of reference answer and model output
    info_list = []
    for gt, pred in tqdm(zip(df_full["Ground Truth"], df_full[args.target_model]), total=len(df_full)):
        output = evaluator(ground_truth_report=gt, predicted_report=pred)
        info_list.append(output)

    # (6) Save the results
    output_dir = args.output_dir

    reasoning_list = []
    score_list     = []
    for i in info_list:
        reasoning_list.append(i['reasoning'])
        score_list.append(i['score'])

    gpt_full = pd.DataFrame()
    gpt_full['reasoning'] = reasoning_list
    gpt_full['score'] = score_list

    # save
    gpt_full.to_csv(output_dir, index=False)



# -----------------------------------------------------------------------------
# 4) Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
