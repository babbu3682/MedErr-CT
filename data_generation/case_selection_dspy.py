"""
Case selection for MedErr-CT (CT-RATE).

    Step 1. Parse the lung section from each full report (LLM).
    Step 2. Label the parsed lung section for lung nodule / pleural effusion (LLM)
            and check it with keyword rules.
    Step 3. Keep a case only when the RadBERT labels, the LLM labels and the keyword rules agree.
    Step 4. Draw a random sample of each group (default 1,000 cases) for error generation.

Models: openai/gpt-4o-mini (default) or ollama_chat/llama3.3:70b-instruct-fp16.

Groups (mutually exclusive):
    nodule   : RadBERT nodule=1, effusion=0 | LLM nodule=1, effusion=0 | lung section contains a nodule keyword
    effusion : RadBERT nodule=0, effusion=1 | LLM nodule=0, effusion=1 | lung section contains an effusion keyword
    none     : RadBERT nodule=0, effusion=0 | LLM nodule=0, effusion=0 | LLM abnormality == 'no findings'

Example:
    python case_selection_dspy.py \
        --report_csv <reports_train.csv> <reports_valid.csv> \
        --label_csv <radbert_predicted_labels.csv> \
        --save_path <output folder> \
        --model openai/gpt-4o-mini --num_threads 4
"""

import pandas as pd
import dspy
import os
import re
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from case_selection_prompt import *


# Keyword rules (lower-case substring match on the parsed lung section)
NODULE_KEYWORDS      = ['nodule']
EFFUSION_KEYWORDS    = ['effusion']
NO_FINDINGS_KEYWORD  = 'no findings'

GROUPS = ['nodule', 'effusion', 'none']


class CaseSelection_Module(dspy.Module):
    def __init__(self):
        super().__init__()
        self.lung_extractor = dspy.ChainOfThought(Extract_LungSection)
        self.lung_labeler   = dspy.ChainOfThought(Label_LungSection)

    def forward(self, report):
        answer_lung  = self.lung_extractor(report=report)
        answer_label = self.lung_labeler(report=answer_lung.lung_section)

        return dspy.Prediction(lung_section=answer_lung.lung_section,
                               lung_nodule=answer_label.lung_nodule,
                               pleural_effusion=answer_label.pleural_effusion,
                               abnormality=answer_label.abnormality)


def build_lm(model, api_base):
    if model.startswith('ollama_chat/'):
        return dspy.LM(model=model, api_base=api_base, model_type='chat', max_tokens=4096, temperature=0.0)

    if model.startswith('openai/'):
        assert os.environ.get('OPENAI_API_KEY'), "Set the OPENAI_API_KEY environment variable."
        return dspy.LM(model, temperature=0.0, max_tokens=4096)

    raise ValueError(f"Unsupported model: {model}")


def parse_binary(value, name):
    value = str(value).strip()
    if value not in ('0', '1'):
        raise ValueError(f"{name} must be 0 or 1, got: {value!r}")
    return int(value)


def normalize(text):
    return re.sub(r'\s+', ' ', str(text)).strip().lower()


def has_keyword(text, keywords):
    text = normalize(text)
    return int(any(keyword in text for keyword in keywords))


def assign_group(row):
    radbert = (row['Lung nodule'], row['Pleural effusion'])
    llm     = (row['llm_lung_nodule'], row['llm_pleural_effusion'])

    if radbert == (1, 0) and llm == (1, 0) and row['rule_nodule'] == 1:
        return 'nodule'
    if radbert == (0, 1) and llm == (0, 1) and row['rule_effusion'] == 1:
        return 'effusion'
    if radbert == (0, 0) and llm == (0, 0) and row['rule_no_findings'] == 1:
        return 'none'
    return 'excluded'


def sample_groups(result_df, n_per_group, seed, save_path):
    """Random sample of each group, saved under the file names that error_generation_dspy.py reads."""
    counts = {group: int((result_df['group'] == group).sum()) for group in GROUPS}
    short = {group: n for group, n in counts.items() if n < n_per_group}
    if short:   # checked for all groups before any file is written
        raise RuntimeError(f"Fewer candidates than --n_per_group {n_per_group}: {short}")
    for group in GROUPS:
        candidates = result_df[result_df['group'] == group]
        sample_df = candidates.sample(n=n_per_group, random_state=seed)
        sample_df.to_csv(f"{save_path}/{group}_sample_{n_per_group}.csv", index=False)
        print(f"{group}: sampled {len(sample_df)} of {len(candidates)} candidates")


def main():
    parser = argparse.ArgumentParser(description='Select nodule / effusion / none cases from CT-RATE reports.')
    parser.add_argument('--report_csv', type=str, nargs='+', required=True, help='CSV file(s) with the report id and the full report.')
    parser.add_argument('--label_csv', type=str, required=True, help='CSV file with the RadBERT labels (VolumeName, Lung nodule, Pleural effusion).')
    parser.add_argument('--save_path', type=str, required=True, help='Directory to save the outputs.')
    parser.add_argument('--id_col', type=str, default='id', help='Column of the report id (CT-RATE volume name).')
    parser.add_argument('--report_col', type=str, default='full_report', help='Column of the full report.')
    parser.add_argument('--model', type=str, default='openai/gpt-4o-mini', help='openai/gpt-4o-mini or ollama_chat/llama3.3:70b-instruct-fp16')
    parser.add_argument('--api_base', type=str, default='http://127.0.0.1:11434', help='API base of the ollama server.')
    parser.add_argument('--num_threads', type=int, default=1, help='Number of concurrent LLM requests.')
    parser.add_argument('--n_per_group', type=int, default=1000, help='Number of cases sampled per group.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed of the sampling.')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Reports
    data_df = pd.concat([pd.read_csv(path) for path in args.report_csv], axis=0).reset_index(drop=True)
    data_df = data_df[[args.id_col, args.report_col]].rename(columns={args.id_col: 'id', args.report_col: 'full_report'})
    assert data_df['id'].is_unique, "Duplicated report ids."
    assert data_df['full_report'].notnull().all(), "Null full reports."

    # RadBERT labels
    label_df = pd.read_csv(args.label_csv)[['VolumeName', 'Lung nodule', 'Pleural effusion']]
    assert label_df['VolumeName'].is_unique, "Duplicated VolumeName in the label CSV."
    assert label_df[['Lung nodule', 'Pleural effusion']].isin([0, 1]).all().all(), "RadBERT labels must be 0 or 1."

    merged_df = data_df.merge(label_df, left_on='id', right_on='VolumeName', how='inner').drop(columns=['VolumeName'])
    print(f"Reports: {len(data_df)}, with RadBERT labels: {len(merged_df)}")
    assert len(merged_df) > 0, "No report matched the label CSV."

    # A case with both labels positive cannot enter any group.
    both = (merged_df['Lung nodule'] == 1) & (merged_df['Pleural effusion'] == 1)
    merged_df = merged_df[~both].reset_index(drop=True)
    print(f"Excluded (RadBERT nodule=1 and effusion=1): {both.sum()}, remaining: {len(merged_df)}")

    print(f"Cases to process: {len(merged_df)}")

    # MODEL
    lm = build_lm(args.model, args.api_base)
    dspy.configure(lm=lm)
    selector = CaseSelection_Module()

    # LLM outputs are appended to a JSONL file, so an interrupted run resumes from it.
    model_tag = re.sub(r'[^A-Za-z0-9.]+', '_', args.model)
    jsonl_path = f"{args.save_path}/case_selection_llm_{model_tag}.jsonl"
    done = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                if record['error'] is None:
                    done[record['id']] = record
    print(f"Already processed: {len(set(done) & set(merged_df['id']))}")

    def process(case_id, report):
        try:
            answer = selector(report=report)
            return {'id': case_id,
                    'lung_section': str(answer.lung_section).strip(),
                    'llm_lung_nodule': parse_binary(answer.lung_nodule, 'lung_nodule'),
                    'llm_pleural_effusion': parse_binary(answer.pleural_effusion, 'pleural_effusion'),
                    'llm_abnormality': str(answer.abnormality).strip(),
                    'error': None}
        except Exception as e:
            return {'id': case_id, 'error': f"{type(e).__name__}: {e}"}

    todo = [(row['id'], row['full_report']) for _, row in merged_df.iterrows() if row['id'] not in done]
    lock = threading.Lock()
    n_error = 0
    with open(jsonl_path, 'a') as f, ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [executor.submit(process, case_id, report) for case_id, report in todo]
        for future in tqdm(as_completed(futures), total=len(futures)):
            record = future.result()
            with lock:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
            if record['error'] is None:
                done[record['id']] = record
            else:
                n_error += 1

    # A failed case is never given a default label. Re-run to retry the failed cases.
    if n_error > 0:
        raise RuntimeError(f"{n_error} cases failed (see 'error' in {jsonl_path}). Re-run the same command to retry them.")

    # Rule-based checks and group assignment
    llm_df = pd.DataFrame([done[case_id] for case_id in merged_df['id']]).drop(columns=['error'])
    result_df = merged_df.merge(llm_df, on='id', how='inner')
    assert len(result_df) == len(merged_df), "LLM outputs are missing for some cases."

    result_df['rule_nodule']      = result_df['lung_section'].apply(lambda x: has_keyword(x, NODULE_KEYWORDS))
    result_df['rule_effusion']    = result_df['lung_section'].apply(lambda x: has_keyword(x, EFFUSION_KEYWORDS))
    result_df['rule_no_findings'] = result_df['llm_abnormality'].apply(lambda x: int(normalize(x).rstrip('.') == NO_FINDINGS_KEYWORD))
    result_df['group'] = result_df.apply(assign_group, axis=1)

    print(result_df['group'].value_counts().to_string())

    result_df.to_csv(f"{args.save_path}/case_selection_{model_tag}.csv", index=False)
    for group in GROUPS:
        result_df[result_df['group'] == group].to_csv(f"{args.save_path}/{group}_candidates_{model_tag}.csv", index=False)

    # Sampling
    sample_groups(result_df, args.n_per_group, args.seed, args.save_path)


if __name__ == '__main__':
    main()
