from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

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

    def encoder(self, target_sentence):
        return self.encoder_model.encode(target_sentence)
    
    def semantic_similarity(self, candidate_sentence: str, reference_sentence: str):
        return 1-cosine(self.encoder(candidate_sentence), self.encoder(reference_sentence))
    
    def semantic_similarity_list(self, candidate_sentence: str, reference_sentences: list):
        encode_cand = self.encoder(candidate_sentence)
        return [1-cosine(encode_cand, self.encoder(item)) for item in reference_sentences]