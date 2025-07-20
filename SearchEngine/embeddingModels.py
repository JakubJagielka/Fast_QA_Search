import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from .utils._logger import get_logger

logger = get_logger()

class MiniLM_L6:
    dimension = 384
    def __init__(self, device=None):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Using only CPU.")
            logger.warning("Are you sure you want to continue? It might be slow for large datasets.")
            input("Press Enter to continue or Ctrl+C to cancel.")
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

    def get_embeddings(self, texts):
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            model_output = self.model(**encoded)
        embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
        
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype(np.float32)
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)