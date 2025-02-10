import torch

def get_words_from_filename(filename) -> str:
    words = set()

    # Remove file extension
    name_without_ext = filename.rsplit('.', 1)[0]
    
    # Replace hyphens and underscores with spaces
    parts = name_without_ext.replace('-', ' ').replace('_', ' ')
    
    # Remove digits
    parts = ''.join(c for c in parts if not c.isdigit())
    
    # Split into words and add to set
    words.update(word.lower() for word in parts.split())

    # Return a string of words separated by spaces (sorted)
    return ' '.join(sorted(words))

def get_embeddings(tokenizer, model, words: str) -> torch.Tensor:
    # Tokenize the input words, padding and truncating as needed
    tokens = tokenizer(words, return_tensors="pt", padding=True, truncation=True)
    # print("tokens : ", tokens)

    # Ensure no gradients are computed
    with torch.no_grad():
        outputs = model(**tokens)

    # Extract token embeddings (squeeze to remove the batch dimension)
    token_embeddings = outputs.last_hidden_state.squeeze(0)

    # print("token_embeddings.size() : ", token_embeddings.size())
    return token_embeddings

def get_top_n_keys(keys_tensor: torch.Tensor, values_tensor: torch.Tensor, n: int = 10) -> torch.Tensor:
    # Ensure that both tensors are of the same length
    assert keys_tensor.size(0) == values_tensor.size(0), "Tensors must have the same length"
    
    # Get the indices that would sort the values_tensor in descending order
    sorted_indices = torch.argsort(values_tensor, descending=True)
    
    # Get the top n indices
    top_n_indices = sorted_indices[:n]
    
    # Use these indices to get the corresponding top n keys
    top_n_keys = keys_tensor[top_n_indices]
    
    return top_n_keys

def similarity_score(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
    # Compute the cosine similarity between the two embeddings
    cosine_similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)

    # print("cosine_similarity : ", cosine_similarity)

    # Return the similarity score
    return cosine_similarity.item()

def are_similar(embedding1: torch.Tensor, embedding2: torch.Tensor, cutoff: float = 0.7) -> bool:
    # Compute the cosine similarity between the two embeddings
    cosine_similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)

    # print("cosine_similarity : ", cosine_similarity)

    # Return True if the similarity is above the cutoff, False otherwise
    return cosine_similarity > cutoff