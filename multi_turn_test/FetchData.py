import json 
import pandas as pd
import copy

class FetchOriginalData:
    def __init__(self):
        print("Loading original data...")
        CHECK_LLM = 0
        all_lms = ["qwen2_0B5", "qwen2_1B5", "qwen2_7B", "mistral03_7B", "llama3_8B", "gemma2_9B"]
        result_403_path = "result/MTMT/{}/MR0_round_original.pickle"
        result_s97_path = "result/MTMT/{}/S97_MR10_SNP_s06_synonym_replacement.pickle"
        df_403 = pd.read_pickle(result_403_path.format(all_lms[CHECK_LLM]))
        df_s97 = pd.read_pickle(result_s97_path.format(all_lms[CHECK_LLM]))
        df_full = pd.concat([df_403, df_s97], axis=0, ignore_index=True)
        df_full["score_semantic"]=[item[0] if item else None for item in df_full["score_semantic"]]
        self.df_full = df_full
        print("Data loaded successfully!")


    def fetch_question(self, dialogue_key, round_key):
        return self.df_full.loc[dialogue_key, round_key]["question"]
    
    def fetch_answer(self, dialogue_key, round_key):
        return self.df_full.loc[dialogue_key, round_key]["answer"]
    
    def fetch_context_sequence(self, dialogue_key, round_key):
        return self.df_full.loc[dialogue_key, round_key]


