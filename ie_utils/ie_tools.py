from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))  # Recursively flatten the list
        else:
            flat_list.append(item)
    return flat_list


class sentence_transformer_tools:
    #encoder_model can be a string or model object, default: SentenceTransformer(encoder_model='sentence-transformers/all-MiniLM-L6-v2', device="cuda")
    def __init__(self, encoder_model='sentence-transformers/all-MiniLM-L6-v2', device="cuda"):
        if type(encoder_model)==type(""):
            self.encoder_model = SentenceTransformer(encoder_model, device=device)
        else:
            self.encoder_model = encoder_model

    def __del__(self):
        try:
            del self.encoder_model
        except:
            None

    def get_model(self):
        return self.encoder_model

    def encoder(self, target_sentence):
        return self.encoder_model.encode(target_sentence)
    
    def semantic_similarity(self, candidate_sentence: str, reference_sentence: str):
        return 1-cosine(self.encoder(candidate_sentence), self.encoder(reference_sentence))
    
    def semantic_similarity_list(self, candidate_sentence: str, reference_sentences: list):
        encode_cand = self.encoder(candidate_sentence)
        return [1-cosine(encode_cand, self.encoder(item)) for item in reference_sentences]


class entity_set_tools:
    def __init__(self, all_entity_types, all_entities):
        self.all_entity_types = all_entity_types
        self.all_entities = all_entities
        self.df_all_entities = pd.DataFrame(all_entities)
        self.entity_group_member_mapping_cache = {}

    def add_new_entity(self, target_entity_name, new_entity_type, new_entity_desc="") -> int:
        self.all_entities.append({"name": target_entity_name, "type":new_entity_type, "description":new_entity_desc})
        # print("before add_new_entity:", target_entity_name,len(self.df_all_entities))
        self.df_all_entities.loc[len(self.df_all_entities)] = {"name": target_entity_name, "type":new_entity_type, "description":new_entity_desc}
        # print("after add_new_entity:", target_entity_name,len(self.df_all_entities))
        return len(self.df_all_entities)

    def get_single_existing_entity_index(self, entity_name) -> int:
        # need to make sure entity_name exists
        if entity_name in self.df_all_entities["name"].to_list():
            return int(self.df_all_entities[self.df_all_entities["name"] == entity_name].index[0])
        else:
            raise(SyntaxError("bad entity_name:{} not in all_entities".format(entity_name)))

    def get_entity_index(self, target_entity_name) -> tuple:
        if target_entity_name in self.df_all_entities["name"].to_list():
            # existing entity
            # print("existing entity")
            return "entity_existing", [self.get_single_existing_entity_index(target_entity_name)]
        
        else:
            # check if in CACHE: entity_group_member_mapping_cache
            if target_entity_name in self.entity_group_member_mapping_cache.keys(): 
                # print("using CACHE for: ",target_entity_name)
                return "entity_group", self.entity_group_member_mapping_cache[target_entity_name]
            
            # not exist, analyse if it is a group of existing entities, if not, make a new one into both list
            # logger.warning("get_entity_index: use LLM for entity: {}".format(target_entity_name))
            dict_subset_analysis = ie_vanilla.entity_subset_filter(entity_list=self.all_entities, entity_types=self.all_entity_types, target_entity_name=target_entity_name)

            # is subset
            if dict_subset_analysis["is_subset_flag"] == 1:
                existing_entity_list = dict_subset_analysis["members"]
                # print("is subset")
                existing_entity_index_list = [self.get_single_existing_entity_index(item["name"]) for item in existing_entity_list]
                # write to cache
                self.entity_group_member_mapping_cache[target_entity_name] = existing_entity_index_list
                # logger.debug("get_entity_index: {} is a subset: {}".format(target_entity_name, list(self.df_all_entities.loc[existing_entity_index_list, "name"])))
                return "entity_group", existing_entity_index_list
            
            # do not exist
            elif dict_subset_analysis["is_subset_flag"] == 0:
                # print("create new: ",target_entity_name)
                return_info = dict_subset_analysis["members"][0]
                self.add_new_entity(target_entity_name=target_entity_name,new_entity_type=return_info["type"])
                # print("create new")
                return "entity_new", [self.get_single_existing_entity_index(target_entity_name)]
            
            # raise error
            else:
                raise SyntaxError("Terrible entity_name, LLM return:{}".format(dict_subset_analysis))
    
    def wash_entity(self,target_entity_name) -> list:
        if target_entity_name:
            if isinstance(target_entity_name, str):
                pass
            elif isinstance(target_entity_name, dict):
                target_entity_name = target_entity_name["name"]
            else:
                raise(SyntaxError("target_entity_name should be string or dict"))
            
            # original form of entity list with all info
            washing_info, clean_entity_index_list = self.get_entity_index(target_entity_name)
            return [self.all_entities[i] for i in clean_entity_index_list]
        else:
            return []

# handle strange relations
class entity_relation_set_tools(entity_set_tools):
    def __init__(self, all_entity_types, all_entities, all_relations, encoder_model='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__(all_entity_types, all_entities)
        self.all_relations = all_relations
        self.df_all_relations = pd.DataFrame(all_relations)

        if isinstance(encoder_model, str):
                self.encoder_model = SentenceTransformer(encoder_model, device="cuda",)
        else:
            self.encoder_model = encoder_model

    def get_relation_index(self, relationship_name=None, source_entity_name=None, target_entity_name=None) -> list:
        # get relation index in df_all_relations, return [] if not exist
        return list(
            self.df_all_relations[
                ((self.df_all_relations["relationship"]==relationship_name) if relationship_name is not None else True)&
                ((self.df_all_relations["source_entity"]==source_entity_name) if source_entity_name is not None else True)&
                ((self.df_all_relations["target_entity"]==target_entity_name) if target_entity_name is not None else True)
            ].index)

    def add_single_new_relation(self, relationship_name, source_entity_name, target_entity_name, relationship_description="") -> int:
        self.all_relations.append({"relationship": relationship_name,"source_entity": source_entity_name,"target_entity": target_entity_name,"relationship_description": relationship_description})
        self.df_all_relations = pd.DataFrame(self.all_relations)
        # print("add new relation: {}".format({"relationship": relationship_name,"source_entity": source_entity_name,"target_entity": target_entity_name,"relationship_description": relationship_description}))
        return self.get_relation_index(relationship_name=relationship_name, source_entity_name=source_entity_name, target_entity_name=target_entity_name)

    def encoder(self, target_sentence):
        return self.encoder_model.encode(target_sentence)
    
    def semantic_similarity(self, candidate_sentence: str, reference_sentence: str):
        return 1-cosine(self.encoder(candidate_sentence), self.encoder(reference_sentence))

    def semantic_similarity_list(self, candidate_sentence: str, reference_sentences: list):
        encode_cand = self.encoder(candidate_sentence)
        return [1-cosine(encode_cand, self.encoder(item)) for item in reference_sentences]
    
    def get_relation_index_similarity_semantic(self, lookingfor_relation_name, all_relations_index_same_source_and_target) -> list:
        df_piece = self.df_all_relations.loc[all_relations_index_same_source_and_target, "relationship"]
        relation_similarity = self.semantic_similarity_list(lookingfor_relation_name, list(df_piece))
        max_i_pos = np.argmax(relation_similarity)
        if relation_similarity[max_i_pos] > 0.8:
            # logger.debug("found similar relation: {} -> {}".format(lookingfor_relation_name, df_piece.iloc[max_i_pos]["relationship"]))
            return all_relations_index_same_source_and_target[max_i_pos]
        else:
            return []

    def single_relation_existence_handler(self, lookingfor_relation_name, source_entity_name, target_entity_name, target_relation_description="") -> list:
        # source_entity and target_entity must exist in all_entities
        # return potential similar relation [index]
        # if no matche, make new relation and return [index]
        # print("single_relation_existence_handler get task: lookingfor_relation_name:{}, source_entity_name:{}, target_entity_name:{}".format(lookingfor_relation_name, source_entity_name, target_entity_name))
        if source_entity_name not in list(self.df_all_entities["name"]):
            raise(SyntaxError("source_entity not in all_entities"))
        
        if target_entity_name not in list(self.df_all_entities["name"]):
            raise(SyntaxError("target_entity not in all_entities"))

        target_relation_index = self.get_relation_index(relationship_name=lookingfor_relation_name, source_entity_name = source_entity_name, target_entity_name=target_entity_name)

        # exist relation between source_entity and target_entity
        if target_relation_index and target_relation_index[0] is not None: 
            # print("single_relation_existence_handler return exist relation: {}".format(target_relation_index))
            # logger.debug("single_relation_existence_handler: existing relation, input:{}, output: {}".format(lookingfor_relation_name, source_entity_name, target_entity_name), target_relation_index)
            if target_relation_index is None: raise(SyntaxError("single_relation_existence_handler: exist relation, but error at target_relation_index"))
            return target_relation_index
        
        # does not exist
        else:
            # get all same source_entity&target_entity relations if any
            all_relations_index_same_source_and_target = self.get_relation_index(source_entity_name=source_entity_name, target_entity_name=target_entity_name)
            
            # try to find the most similar one
            if all_relations_index_same_source_and_target:
                most_similar_relation_index = self.get_relation_index_similarity_semantic(lookingfor_relation_name, all_relations_index_same_source_and_target)
                if most_similar_relation_index:
                    # print("single_relation_existence_handler return most similar relation: {}".format(most_similar_relation_index))
                    if target_relation_index is None: raise(SyntaxError("single_relation_existence_handler: Found similar, but error at target_relation_index"))
                    return most_similar_relation_index

                else:
                    res =  self.add_single_new_relation(relationship_name=lookingfor_relation_name, source_entity_name=source_entity_name, target_entity_name=target_entity_name, relationship_description=target_relation_description)
                    if res is None: raise(SyntaxError("single_relation_existence_handler: make new relation, Error at target_relation_index"))
                    return res
                
            # make new relation: no similar connect between source_entity_name and target_entity_name, or no matching relation
            else: 
                res =  self.add_single_new_relation(relationship_name=lookingfor_relation_name, source_entity_name=source_entity_name, target_entity_name=target_entity_name, relationship_description=target_relation_description)
                if target_relation_index is None: raise(SyntaxError("single_relation_existence_handler: make new relation, Error at target_relation_index"))
                return res
        

    def complex_relation_handler(self, relationship_name, source_entity_name, target_entity_name, target_relation_description="") -> list:
        # source_entity, target_entity can be entity group or new entity
        # will connect between all members in source_entity and target_entity with relationship_name
        source_entity_info, source_entity_ids = self.get_entity_index(source_entity_name)
        target_entity_info, target_entity_ids = self.get_entity_index(target_entity_name)
        # print("complex_relation_handler get relation:{}, source_entity_ids: {} and source_entity_ids:{}".format(relationship_name, source_entity_ids, target_entity_ids))
        new_relation_index=[]
        for source_entity_id in source_entity_ids:
            for target_entity_id in target_entity_ids:
                this_relation_index = self.single_relation_existence_handler(lookingfor_relation_name=relationship_name, source_entity_name=self.all_entities[source_entity_id]["name"], target_entity_name=self.all_entities[target_entity_id]["name"], target_relation_description=target_relation_description)
                
                if this_relation_index is None:
                    raise(SyntaxError("Something ain't right, single_relation_existence_handler return: None, function input:{lookingfor_relation_name}, {source_entity_name}, {target_entity_name}".format(lookingfor_relation_name=relationship_name, source_entity_name=self.all_entities[source_entity_id]["name"], target_entity_name=self.all_entities[target_entity_id]["name"])))
                new_relation_index.append(this_relation_index)
                # logger.debug("loop in complex_relation_handler, input:{}, output: {}".format([source_entity_id, target_entity_id], this_relation_index))

        # logger.debug("complex_relation_handler, input:{}, output: {}".format([relationship_name, source_entity_name, target_entity_name],new_relation_index))
        return flatten(new_relation_index)
    
    def wash_relation(self, target_relation_dict) -> list:
        if target_relation_dict:
            relationship_name = target_relation_dict["relationship"]
            source_entity_name = target_relation_dict["source_entity"]
            target_entity_name = target_relation_dict["target_entity"]
            target_relation_description = target_relation_dict["relationship_description"] if "relationship_description" in target_relation_dict.keys() else ""
            new_relation_index = self.complex_relation_handler(relationship_name=relationship_name, source_entity_name=source_entity_name, target_entity_name=target_entity_name, target_relation_description=target_relation_description)
            return [self.all_relations[i] for i in new_relation_index]
        else:
            return []
        