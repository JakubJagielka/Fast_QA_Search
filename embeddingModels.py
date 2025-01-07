from transformers import AutoTokenizer, AutoModel
import torch

class MiniLM_L6:
    dimension = 384
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)


    def get_embeddings(self, texts):
        # Tokenize the input texts
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # Get model output
        with torch.no_grad():
            model_output = self.model(**encoded)
        # Mean pooling - take mean of all tokens
        embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
