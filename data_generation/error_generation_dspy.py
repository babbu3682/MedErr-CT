import pandas as pd
import dspy
import os
import random
import argparse
from tqdm import tqdm
from error_generation_prompt import *
from insertion_template import nodule_insert_template, effusion_insert_template





def main():
    parser = argparse.ArgumentParser(description='Process NIfTI image and encode it using a transformer model.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the NIfTI image file.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the encoded image.')
    parser.add_argument('--start_idx', type=int, default=None, help='Start index for processing. train=0')
    parser.add_argument('--end_idx', type=int, default=None, help='End index for processing. train=18607')
    args = parser.parse_args()

    # CSV path
    nodule_df   = pd.read_csv(f"{args.csv_path}/nodule_sample_1000.csv")
    effusion_df = pd.read_csv(f"{args.csv_path}/effusion_sample_1000.csv")
    none_df     = pd.read_csv(f"{args.csv_path}/none_sample_1000.csv")

    # MODEL
    lm = dspy.LM(
            model='ollama_chat/llama3.3:70b-instruct-fp16',
            api_base="http://127.0.0.1:11434", 
            model_type='chat',
            max_tokens=4096,
            temperature=0.0)

    dspy.configure(lm=lm)

    # Dspy module define
    omission_nodule    = dspy.ChainOfThought(DistortReport_Omission_Nodule)
    omission_effusion  = dspy.ChainOfThought(DistortReport_Omission_Effusion)

    insertion_nodule   = dspy.ChainOfThought(DistortReport_Insertion_Nodule) 
    insertion_effusion = dspy.ChainOfThought(DistortReport_Insertion_Effusion)

    direction_nodule   = dspy.ChainOfThought(DistortReport_Direction_Nodule)
    direction_effusion = dspy.ChainOfThought(DistortReport_Direction_Effusion)

    size = dspy.ChainOfThought(DistortReport_Size) 
    typo = dspy.ChainOfThought(DistortReport_Typo)
    unit = dspy.ChainOfThought(DistortReport_Unit)

    # Nodule
    for idx, row in tqdm(nodule_df.iterrows()):
        
        # Omission
        nodule_df.loc[idx, 'omission_nodule_distorted_report'] = omission_nodule(report=row['lung_parenchyma']).distorted_report
        nodule_df.loc[idx, 'omission_nodule_classification'] = omission_nodule(report=row['lung_parenchyma']).classification
        nodule_df.loc[idx, 'omission_nodule_deleted_sentence'] = omission_nodule(report=row['lung_parenchyma']).deleted_finding
        
        # Insertion
        insert_sentence_effusion = random.choice(effusion_insert_template)
        nodule_df.loc[idx, 'insertion_effusion_distorted_report'] = insertion_effusion(report=row['lung_parenchyma'], insert_sentence=insert_sentence_effusion).distorted_report
        nodule_df.loc[idx, 'insertion_effusion_classification'] = insertion_effusion(report=row['lung_parenchyma'], insert_sentence=insert_sentence_effusion).classification
        nodule_df.loc[idx, 'insertion_effusion_inserted_sentence'] = insert_sentence_effusion
        
        # Direction
        nodule_df.loc[idx, 'direction_nodule_distorted_report'] = direction_nodule(report=row['lung_parenchyma']).distorted_report
        nodule_df.loc[idx, 'direction_nodule_classification'] = direction_nodule(report=row['lung_parenchyma']).classification
        nodule_df.loc[idx, 'direction_nodule_distorted_sentence'] = direction_nodule(report=row['lung_parenchyma']).distorted_sentence
        nodule_df.loc[idx, 'direction_nodule_corrected_sentence'] = direction_nodule(report=row['lung_parenchyma']).corrected_sentence
        
        # Size
        nodule_df.loc[idx, 'size_distorted_report'] = size(report=row['lung_parenchyma']).distorted_report
        nodule_df.loc[idx, 'size_classification'] = size(report=row['lung_parenchyma']).classification
        nodule_df.loc[idx, 'size_distorted_sentence'] = size(report=row['lung_parenchyma']).distorted_sentence
        nodule_df.loc[idx, 'size_corrected_sentence'] = size(report=row['lung_parenchyma']).corrected_sentence
        nodule_df.loc[idx, 'size_disease_selected'] = size(report=row['lung_parenchyma']).disease_selected
        
        # Typo
        nodule_df.loc[idx, 'typo_distorted_report'] = typo(report=row['lung_parenchyma']).distorted_report
        nodule_df.loc[idx, 'typo_classification'] = typo(report=row['lung_parenchyma']).classification
        nodule_df.loc[idx, 'typo_distorted_sentence'] = typo(report=row['lung_parenchyma']).distorted_sentence
        nodule_df.loc[idx, 'typo_corrected_sentence'] = typo(report=row['lung_parenchyma']).corrected_sentence

        # Unit
        nodule_df.loc[idx, 'unit_distorted_report'] = unit(report=row['lung_parenchyma']).distorted_report
        nodule_df.loc[idx, 'unit_classification'] = unit(report=row['lung_parenchyma']).classification
        nodule_df.loc[idx, 'unit_distorted_sentence'] = unit(report=row['lung_parenchyma']).distorted_sentence
        nodule_df.loc[idx, 'unit_corrected_sentence'] = unit(report=row['lung_parenchyma']).corrected_sentence 
        nodule_df.loc[idx, 'unit_disease_selected'] = unit(report=row['lung_parenchyma']).disease_selected   
    
    nodule_df.to_csv(f"{args.save_path}/dspy_nodule.csv", index=False)

    # Effusion
    for idx, row in tqdm(effusion_df.iterrows()):
        # Omission
        effusion_df.loc[idx, 'omission_effusion_distorted_report'] = omission_effusion(report=row['lung_parenchyma']).distorted_report
        effusion_df.loc[idx, 'omission_effusion_classification'] = omission_effusion(report=row['lung_parenchyma']).classification
        effusion_df.loc[idx, 'omission_effusion_deleted_sentence'] = omission_effusion(report=row['lung_parenchyma']).deleted_finding

        # Insertion
        insert_sentence_nodule = random.choice(nodule_insert_template)
        effusion_df.loc[idx, 'insertion_nodule_distorted_report'] = insertion_nodule(report=row['lung_parenchyma'], insert_sentence=insert_sentence_nodule).distorted_report
        effusion_df.loc[idx, 'insertion_nodule_classification'] = insertion_nodule(report=row['lung_parenchyma'], insert_sentence=insert_sentence_nodule).classification
        effusion_df.loc[idx, 'insertion_nodule_inserted_sentence'] = insert_sentence_nodule    
        
        # Direction
        effusion_df.loc[idx, 'direction_effusion_distorted_report'] = direction_effusion(report=row['lung_parenchyma']).distorted_report
        effusion_df.loc[idx, 'direction_effusion_classification'] = direction_effusion(report=row['lung_parenchyma']).classification
        effusion_df.loc[idx, 'direction_effusion_distorted_sentence'] = direction_effusion(report=row['lung_parenchyma']).distorted_sentence
        effusion_df.loc[idx, 'direction_effusion_corrected_sentence'] = direction_effusion(report=row['lung_parenchyma']).corrected_sentence
        
        # Size
        effusion_df.loc[idx, 'size_distorted_report'] = size(report=row['lung_parenchyma']).distorted_report
        effusion_df.loc[idx, 'size_classification'] = size(report=row['lung_parenchyma']).classification
        effusion_df.loc[idx, 'size_distorted_sentence'] = size(report=row['lung_parenchyma']).distorted_sentence
        effusion_df.loc[idx, 'size_corrected_sentence'] = size(report=row['lung_parenchyma']).corrected_sentence
        effusion_df.loc[idx, 'size_disease_selected'] = size(report=row['lung_parenchyma']).disease_selected
        
        # Typo
        effusion_df.loc[idx, 'typo_distorted_report'] = typo(report=row['lung_parenchyma']).distorted_report
        effusion_df.loc[idx, 'typo_classification'] = typo(report=row['lung_parenchyma']).classification
        effusion_df.loc[idx, 'typo_distorted_sentence'] = typo(report=row['lung_parenchyma']).distorted_sentence
        effusion_df.loc[idx, 'typo_corrected_sentence'] = typo(report=row['lung_parenchyma']).corrected_sentence
        
        # Unit
        effusion_df.loc[idx, 'unit_distorted_report'] = unit(report=row['lung_parenchyma']).distorted_report
        effusion_df.loc[idx, 'unit_classification'] = unit(report=row['lung_parenchyma']).classification
        effusion_df.loc[idx, 'unit_distorted_sentence'] = unit(report=row['lung_parenchyma']).distorted_sentence
        effusion_df.loc[idx, 'unit_corrected_sentence'] = unit(report=row['lung_parenchyma']).corrected_sentence
        effusion_df.loc[idx, 'unit_disease_selected'] = unit(report=row['lung_parenchyma']).disease_selected

    effusion_df.to_csv(f"{args.save_path}/dspy_effusion.csv", index=False)

    # Normal
    for idx, row in tqdm(none_df.iterrows()):
        # Insertion
        insert_sentence_nodule = random.choice(nodule_insert_template)
        none_df.loc[idx, 'insertion_nodule_distorted_report'] = insertion_nodule(report=row['lung_parenchyma'], insert_sentence=insert_sentence_nodule).distorted_report
        none_df.loc[idx, 'insertion_nodule_classification'] = insertion_nodule(report=row['lung_parenchyma'], insert_sentence=insert_sentence_nodule).classification
        none_df.loc[idx, 'insertion_nodule_inserted_sentence'] = insert_sentence_nodule
        
        insert_sentence_effusion = random.choice(effusion_insert_template)
        none_df.loc[idx, 'insertion_effusion_distorted_report'] = insertion_effusion(report=row['lung_parenchyma'], insert_sentence=insert_sentence_effusion).distorted_report
        none_df.loc[idx, 'insertion_effusion_classification'] = insertion_effusion(report=row['lung_parenchyma'], insert_sentence=insert_sentence_effusion).classification
        none_df.loc[idx, 'insertion_effusion_inserted_sentence'] = insert_sentence_effusion
        
    none_df.to_csv(f"{args.save_path}/dspy_none.csv", index=False)

if __name__ == '__main__':
    main()