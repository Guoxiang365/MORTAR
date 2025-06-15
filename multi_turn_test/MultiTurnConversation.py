from transformers import pipeline
import pandas as pd
import torch

import numpy as np

import re
import string
import tqdm

from collections import Counter, OrderedDict

import torch 
from torch import nn

from sentence_transformers import SentenceTransformer


class score_utils:
    #encoder_model can be a string or model object, default: SentenceTransformer(encoder_model='sentence-transformers/all-MiniLM-L6-v2', device="cuda")
    def __init__(self, encoder_model='sentence-transformers/all-MiniLM-L6-v2', device="cuda", use_model=True):
        self.cos_sim = nn.CosineSimilarity(dim=0)
        if use_model:
            if type(encoder_model)==type(""):
                self.encoder_model = SentenceTransformer(encoder_model, device=device)
            else:
                self.encoder_model = encoder_model


    def __del__(self):
        try:
            del self.encoder_model
        except:
            None
    
    def cosine_similarity(self, a: torch.Tensor, b:torch.Tensor):
        return float(self.cos_sim(torch.tensor(a), torch.tensor(b)))

    # refered coqa evaluation code v1.0
    def normalize_answer(self,sentence):
        """Lower text and remove punctuation, storys and extra whitespace."""
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(sentence))))

    def get_tokens(self,s):
        if not s: return []
        return self.normalize_answer(s).split()
    
    def compute_exact(self,a_gold, a_pred):
        return int(self.normalize_answer(a_gold) == self.normalize_answer(a_pred))

    def compute_f1(self,a_gold, a_pred):
        gold_toks = self.get_tokens(a_gold)
        pred_toks = self.get_tokens(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def encoder(self, target_sentence):
        return self.encoder_model.encode(target_sentence)
    
    def semantic_similarity(self, candidate_sentence: str, reference_sentence: str):
        return self.cosine_similarity(self.encoder(candidate_sentence), self.encoder(reference_sentence))
    
    def semantic_similarity_list(self, candidate_sentence: str, reference_sentences: list):
        encode_cand = self.encoder(candidate_sentence)
        return [self.cosine_similarity(encode_cand, self.encoder(item)) for item in reference_sentences]
    
    def compute_mss(self, candidate_sentence: str, reference_sentence: str):
        sbert_semantic_similarity = self.semantic_similarity(candidate_sentence, reference_sentence)
        f1_score = self.compute_f1(candidate_sentence, reference_sentence)
        em_score = self.compute_exact(candidate_sentence, reference_sentence)
        return (sbert_semantic_similarity*sbert_semantic_similarity + f1_score*f1_score + em_score*em_score) / (f1_score + em_score + sbert_semantic_similarity)


class multi_turn_conversation:
    def __init__(self, f_llm_pipeline: pipeline, pipeline_params: dict, score_encode_model='sentence-transformers/all-MiniLM-L6-v2'):
        """
        if score_encode_model is provided as str, load that model
        if score_encode_model is provided as an score_utils instance, use it directly, might improve speed
        """
        self.start_time=pd.Timestamp.now()
        self.f_chat_pipeline = f_llm_pipeline
        self.chat_pipeline_params = pipeline_params
        self.members = ["system","user","assistant"]
        # expected_answer is a list
        # note is a dict
        self.history = pd.DataFrame(columns=["send_time","role","content","expected_answer","source_uuid","round","attachment","note"])
        self.raw_history=[]
        self.message_counter=0
        self.round_counter=0
        self.score_tool = score_utils(score_encode_model) if type(score_encode_model) == type(" ") else score_encode_model
        # self.sending_flag=False

    def temporary_message(self, message_body: str, attachment_list=[]):
        output = self.f_chat_pipeline(
            [{'role': 'user','content': '{}'.format(message_body)}],
            **self.chat_pipeline_params
        )
        return output[0]["generated_text"][-1]["content"]

    def history_append(self, role: str, message_body: str, expected_answer=None, attachment_list=[], source_uuid=None, note=None):
        if role=="user":
            self.round_counter+=1
            
        self.history.loc[self.message_counter] = {
            "send_time": pd.Timestamp.now(),
            "role": role,
            "content": message_body,
            "expected_answer": expected_answer,
            "source_uuid": source_uuid,
            "round": self.round_counter,
            "attachment": attachment_list,
            "note": note,
        }
        self.raw_history.append({
            'role': role, 
            'content': message_body,
            })
        self.message_counter+=1
        return 0
    
    def llm_inference(self):
        output = self.f_chat_pipeline(
            self.raw_history,
            **self.chat_pipeline_params
        )
        llm_said =  output[0]["generated_text"][-1]["content"]
        return llm_said

    def new_rounds(self, message_body: str, expected_answer=None, role="user", attachment_list=[], source_uuid=None, note=None, send_to_lm=True):
        if role not in self.members:
            raise(ValueError("Incorrect role"))
        self.history_append(role=role, message_body=message_body, expected_answer=expected_answer, attachment_list=attachment_list, source_uuid=source_uuid, note=note)
        if role in ["system", "assistant"]:
            send_to_lm = False
        if send_to_lm:
            return_message = self.llm_inference()
            self.history_append(role="assistant",message_body=return_message,expected_answer=expected_answer,source_uuid=source_uuid,note=note)
            return return_message
        return None
    
    def recall_last_message(self):
        to_delete_list = self.history[self.history["round"]==self.round_counter].index
        self.history=self.history.drop(to_delete_list)
        self.round_counter-=1
        self.message_counter-=len(to_delete_list)
        self.raw_history = self.raw_history[:-len(to_delete_list)]
    
    def load_from_history(self, chat_history: pd.DataFrame, raw_history:list):
        self.history = chat_history.copy()
        self.raw_history=raw_history.copy()
        self.message_counter=len(raw_history)

    def scorer(self, gold_text, input_text, mode="f1") -> float:
        if mode=="f1":
            return self.score_tool.compute_f1(gold_text, input_text)
        elif mode=="em":
            return self.score_tool.compute_exact(gold_text, input_text)
        elif mode=="semantic":
            return self.score_tool.semantic_similarity(gold_text, input_text)
    
    def scorer_batch(self, gold_text_list, input_text, mode="f1") -> list:
        """
        mode: "f1", "em", "semantic"
        """
        # print(gold_text_list, input_text)
        if mode=="f1":
            return [self.score_tool.compute_f1(gold_text, input_text) for gold_text in gold_text_list]
        elif mode=="em":
            return [self.score_tool.compute_exact(gold_text, input_text) for gold_text in gold_text_list]
        elif mode=="semantic":
            return self.score_tool.semantic_similarity_list(input_text, gold_text_list)

    def calculate_score(self):
        col_score_em = []
        col_score_f1 = []
        col_score_semantic = []

        for i_row in self.history.index:
            if self.history.loc[i_row, "role"]=="assistant":
                exp_ans = self.history.loc[i_row, "expected_answer"]
                gen_ans = self.history.loc[i_row, "content"]

                f1_score = self.scorer_batch(exp_ans, gen_ans, "f1")
                em_score = self.scorer_batch(exp_ans, gen_ans, "em")
                semantic_score = self.scorer_batch(exp_ans, gen_ans, "semantic")

                col_score_em.append(em_score)
                col_score_f1.append(f1_score)
                col_score_semantic.append(semantic_score)
            else:
                col_score_em.append(None)
                col_score_f1.append(None)
                col_score_semantic.append(None)

        self.history["score_em"] = col_score_em
        self.history["score_f1"] = col_score_f1
        self.history["score_semantic"] = col_score_semantic

    def end_chat(self, evaluate=False):
        if evaluate:
            self.calculate_score()
        del self.f_chat_pipeline
        return self.history
    
if __name__ == "__main__":
    pipe_qwen = pipeline(
        "text-generation", 
        model="Qwen/Qwen2-1.5B-Instruct",
        device="cuda",)

    pipe_qwen_config = {
        "max_new_tokens":256,
        "do_sample":True,
        "temperature":0.6,
        "top_p":0.9,
    }

    mtc_test = multi_turn_conversation(f_llm_pipeline=pipe_qwen, pipeline_params=pipe_qwen_config)
    print(mtc_test.new_rounds("Hi there!", expected_answer=["Hi, I am a AI model", "Hi"]))
    print(mtc_test.new_rounds("Please remember, I have an apple now.", expected_answer=["OK.", "I know."]))
    print(mtc_test.new_rounds("WHat did I say I have just now?", expected_answer=["An apple.", "You said you have an apple."]))

    df_chat_history = mtc_test.end_chat(evaluate=True)
    del pipe_qwen
    torch.cuda.empty_cache()

    # df_chat_history.to_csv("test_output.csv")

