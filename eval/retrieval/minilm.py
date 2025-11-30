import numpy as np
from typing import List, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from eval.retrieval.kv_store import KVStore
from eval.retrieval.kv_store import TextType

class MiniLM(KVStore):
    def __init__(self, index_name: str, model_path: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"):
        """
        Initializes the MiniLM indexer using the sentence-transformers library.
        """
        super().__init__(index_name, 'minilm')
        self.model_path = model_path
        # Initialize the SentenceTransformer model
        self._model = SentenceTransformer(self.model_path, device="cuda") # Assumes CUDA is available
    
    # Note: No _get_instruction method is needed for this model.
    
    def _encode_batch(self, texts: List[str], type: TextType, show_progress_bar: bool = True) -> List[Any]:
        """
        Encodes a batch of texts using the SentenceTransformer model.
        The 'type' parameter is unused here but kept for interface consistency.
        """
        # The encode method is slightly different: no 'instruction' parameter.
        embeddings = self._model.encode(
            texts, 
            batch_size=256, 
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float16) # Convert to float16 for memory efficiency
    
    def _query(self, encoded_query: Any, n: int) -> List[int]:
        """
        Performs cosine similarity search. This method is model-agnostic and
        can be copied directly from your GRIT class.
        """
        try:
            cosine_similarities = cosine_similarity([encoded_query], self.encoded_keys)[0]
        except:
            # Handle potential NaN values in keys
            for i, encoded_key in enumerate(self.encoded_keys):
                if np.any(np.isnan(encoded_key)):
                    self.encoded_keys[i] = np.zeros_like(encoded_key)
            cosine_similarities = cosine_similarity([encoded_query], self.encoded_keys)[0]
            
        top_indices = cosine_similarities.argsort()[-n:][::-1]
        return top_indices.tolist() # Ensure it returns a standard list
    
    def load(self, path: str):
        """
        Loads the index from disk and re-initializes the model for querying.
        """
        super().load(path)
        # Re-initialize the model when loading the index for querying
        self._model = SentenceTransformer(self.model_path, device="cuda")
        return self