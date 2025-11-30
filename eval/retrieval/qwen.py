import numpy as np
from typing import List, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from utils import utils
from eval.retrieval.kv_store import KVStore, TextType

class Qwen3Embedding(KVStore):
    """
    Retrieval index for Qwen/Qwen3-Embedding-* models.
    - Queries get an instruction prefix; passages/keys are raw text.
    - Left padding is recommended for last-token pooling.
    """
    def __init__(
        self,
        index_name: str,
        query_instruction: str = "Given a search query, retrieve relevant passages that answer the query.",
        model_path: str = "Qwen/Qwen3-Embedding-8B",
        batch_size: int = 32,
        use_flash_attn: bool = True,
        dtype: str = "bf16",  # "auto" | "fp16" | "bf16"
    ):
        super().__init__(index_name, "qwen3")
        self.model_path = model_path
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.use_flash_attn = use_flash_attn
        self.dtype = dtype

        model_kwargs = {"device_map": "auto", "trust_remote_code": True}
        # use the new `dtype` kw
        if self.dtype == "fp16":
            model_kwargs["dtype"] = "float16"
        elif self.dtype == "bf16":
            model_kwargs["dtype"] = "bfloat16"
        # enable FA2 only in half precision
        if self.use_flash_attn and model_kwargs.get("dtype") in ("float16", "bfloat16"):
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._model = SentenceTransformer(
            self.model_path,
            cache_folder=utils.get_cache_dir(),
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"padding_side": "left", "trust_remote_code": True},
        )

    def _format_text(self, text: str, type: TextType) -> str:
        if type == TextType.KEY:
            return text
        elif type == TextType.QUERY:
            return f"Instruct: {self.query_instruction}\nQuery: {text}" if self.query_instruction.strip() else text
        else:
            raise ValueError("Invalid TextType")

    def _encode_batch(
        self, texts: List[str], type: TextType, show_progress_bar: bool = True
    ) -> List[Any]:
        texts = [self._format_text(t, type) for t in texts]
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )
        return emb.astype(np.float16)

    def _query(self, encoded_query: Any, n: int) -> List[int]:
        sims = cosine_similarity([encoded_query], self.encoded_keys)[0]
        return sims.argsort()[-n:][::-1]

    def load(self, path: str):
        super().load(path)

        # Rehydrate missing attrs for old pickles
        self.model_path = getattr(self, "model_path", "Qwen/Qwen3-Embedding-8B")
        self.query_instruction = getattr(self, "query_instruction", "Given a search query, retrieve relevant passages that answer the query.")
        self.batch_size = getattr(self, "batch_size", 32)
        self.use_flash_attn = getattr(self, "use_flash_attn", True)
        self.dtype = getattr(self, "dtype", "bf16")

        model_kwargs = {"device_map": "auto", "trust_remote_code": True}
        if self.dtype == "fp16":
            model_kwargs["dtype"] = "float16"
        elif self.dtype == "bf16":
            model_kwargs["dtype"] = "bfloat16"
        if self.use_flash_attn and model_kwargs.get("dtype") in ("float16", "bfloat16"):
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._model = SentenceTransformer(
            self.model_path,
            cache_folder=utils.get_cache_dir(),
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"padding_side": "left", "trust_remote_code": True},
        )
        return self
