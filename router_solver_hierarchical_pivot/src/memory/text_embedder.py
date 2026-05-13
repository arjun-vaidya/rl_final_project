import json
import os
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class RetrievalTextEmbedder:
    _MODEL_CACHE = {}

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu", max_length: int = 256):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.query_prefix = ""
        self.doc_prefix = ""
        self.projector = None
        resolved_model_name = model_name
        projector_state_path = None
        projector_dim = None

        config_path = os.path.join(model_name, "retriever_config.json")
        if os.path.isdir(model_name) and os.path.exists(config_path):
            with open(config_path, "r", encoding="ascii", errors="ignore") as f:
                cfg = json.load(f)
            resolved_model_name = cfg["backbone_model_name"]
            self.query_prefix = str(cfg.get("query_prefix", ""))
            self.doc_prefix = str(cfg.get("doc_prefix", ""))
            projector_state_path = os.path.join(model_name, "projector.pt")
            projector_dim = int(cfg.get("projection_dim", 0) or 0)

        cache_key: Tuple[str, str] = (resolved_model_name, device)
        if cache_key not in self._MODEL_CACHE:
            tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)
            model = AutoModel.from_pretrained(resolved_model_name)
            model.to(device)
            model.eval()
            self._MODEL_CACHE[cache_key] = (tokenizer, model)
        self.tokenizer, self.model = self._MODEL_CACHE[cache_key]
        hidden_size = int(getattr(self.model.config, "hidden_size"))
        self.output_dim = hidden_size

        if not self.query_prefix and not self.doc_prefix and "e5" in resolved_model_name.lower():
            self.query_prefix = "query: "
            self.doc_prefix = "passage: "

        if projector_state_path and os.path.exists(projector_state_path):
            if projector_dim <= 0:
                raise ValueError(f"Invalid projector_dim in {config_path}")
            self.projector = nn.Linear(hidden_size, projector_dim, bias=False)
            state = torch.load(projector_state_path, map_location="cpu")
            self.projector.load_state_dict(state)
            self.projector.to(device)
            self.projector.eval()
            self.output_dim = projector_dim

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], role: str = "query") -> torch.Tensor:
        prefix = self.query_prefix if role == "query" else self.doc_prefix
        texts = [prefix + str(text or "") for text in texts]
        if not texts:
            return torch.empty(0, self.output_dim, dtype=torch.float32)
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        token_embeddings = outputs.last_hidden_state
        attention_mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1)
        if self.projector is not None:
            pooled = self.projector(pooled)
        pooled = F.normalize(pooled, dim=1)
        return pooled.detach().cpu()

    def encode_one(self, text: str, role: str = "query") -> torch.Tensor:
        return self.encode([text], role=role)[0]
