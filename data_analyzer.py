from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List
import os
import torch
from transformers import AutoTokenizer, AutoModel

import utils
from llm_client import LLMClient

# Define an abstract class
class DataAnalyzer(ABC):
    def __init__(self, datapath="./data", model_name="anthropic"):
        # Create an empty dictionary or list to store DataFrames
        self._dataset = {}  # Dictionary to store DataFrames with filenames as keys

        # Loop through the files in the folder
        for filename in os.listdir(datapath):
            match filename:
                case s if s.endswith('.csv'):
                    # Construct full file path
                    file_path = os.path.join(datapath, filename)
                    
                    # Read the CSV into a DataFrame
                    df = pd.read_csv(file_path)
                    
                    # Store the DataFrame in the dictionary with the filename as key
                    self._dataset[filename] = df
                case s if s.endswith('.xlsx'):
                    # Construct full file path
                    file_path = os.path.join(datapath, filename)
                    
                    # Read the Excel file into a DataFrame
                    df = pd.read_excel(file_path)
                    
                    # Store the DataFrame in the dictionary with the filename as key
                    self._dataset[filename] = df
                case _:
                    raise Exception("Unrecognized file format in dataset folder")

        # Now, df_dict contains all your DataFrames, indexed by filenames

        self._client = LLMClient(model=model_name)

    @property
    def dataset(self):
        return self._dataset
    
    @property
    def client(self):
        return self._client

    def answerQuestion(self, messages, question: str) -> str:
        keywords = self.keywords(question) # Type will depend on implementation
        relevantDocuments = self.findRelevantDocuments(keywords)
        retrievedData = []
        for file in relevantDocuments:
            retrievedData.append(" The following data is also useful: " + self.retrieveData(keywords, file))
        retrievedData_str = ''.join(retrievedData)
        newMessage = "The user would like to know the following: " + question + retrievedData_str
        messages.append({"role": "user", "content": newMessage})
        print("newMessage = " + newMessage)
        response = self.client.chat(messages)
        messages.append({"role": "assistant", "content": response})
        return response

    @abstractmethod
    def keywords(self, question: str): # Output type will depend on implementation
        pass

    @abstractmethod
    def findRelevantDocuments(self, keywords) -> List[pd.DataFrame]: # Keywords type will depend on implementation
        pass

    @abstractmethod
    def retrieveData(self, keywords, dataset: pd.DataFrame) -> str: # Keywords type will depend on implementation
        pass

# MVP implementation of the abstract class
class MvpDataAnalyzer(DataAnalyzer):
    # __init__ handled by superclass
    
    def keywords(self, question: str) -> Dict[str, float]:
        # This is a very basic implementation of keyword extraction
        return {word:1.0 for word in question.split() if len(word) > 3} 
    
    def findRelevantDocuments(self, keywords: Dict[str, float]) -> List[pd.DataFrame]:
        relevant_files = []
        for filename, file in self.dataset.items():
            # Check if the file title contains the keywords - this is a very basic check
            filewords = utils.get_words_from_filename(filename)
            if any(fileword == word for word in keywords for fileword in filewords):
                relevant_files.append(file)
        return relevant_files

    def retrieveData(self, keywords: Dict[str, float], df: pd.DataFrame) -> str:
        masks = []
        
        # Get all columns that contain strings (object or string dtype)
        string_columns = df.select_dtypes(include=['object', 'string']).columns
        
        # Create mask for each text column
        for col in string_columns:
            mask = df[col].astype(str).str.contains('|'.join(keywords), case=False, na=False)
            masks.append(mask)
            
        # Combine all masks with OR operation
        if masks:
            final_mask = pd.concat(masks, axis=1).any(axis=1)
            return df[final_mask].to_string(index=False)
            
        return "no data" # Return this message if no data is found
    
# A more complete implementation of the abstract class
class CompleteDataAnalyzer(DataAnalyzer):
    def __init__(self, datapath="./data", model_name="anthropic"):
        super().__init__(datapath, model_name)
        self._tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self._model = AutoModel.from_pretrained("bert-base-uncased")
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def model(self):
        return self._model

    def keywords(self, question: str, word_limit: int = 10) -> List[torch.Tensor]:
        token_embeddings = utils.get_embeddings(self.tokenizer, self.model, question)

        # Compute importance scores as the norm of embeddings for each token
        # This is a simple way to get token importance based on the embedding norm, a better way would be to use attention weights
        token_scores = token_embeddings.norm(dim=1)

        important_tokens = utils.get_top_n_keys(token_embeddings, token_scores, n = word_limit)

        # Return the top 'word_limit' keywords (their embeddings and scores)
        return important_tokens
    
    # Currently, findRelevantDocuments only looks at filenames. An even better approach would be to check the content of the file, the idea I had was to pre-process the data by clustering the embeddings of the text in the file to get a small list of representative embeddings for each file and then compare these embeddings to the keyword embeddings. This would be more efficient than comparing each word embedding in the file to each keyword embedding. I didn't have time to implement this, but I think it would be a good approach.

    def findRelevantDocuments(self, keywords: List[torch.Tensor], top_n: int = 3) -> List[pd.DataFrame]:
        file_scores = []
        for filename, file in self.dataset.items():
            filewords = utils.get_words_from_filename(filename)
            filename_embeddings = utils.get_embeddings(self.tokenizer, self.model, filewords)
            
            max_score = 0
            for keyword_embedding in keywords:
                for filename_embedding in filename_embeddings:
                    score = utils.similarity_score(keyword_embedding, filename_embedding)
                    if score > max_score:
                        max_score = score
            
            file_scores.append((max_score, file))
        
        # Sort the files by their max similarity score and select the top_n files
        file_scores.sort(reverse=True, key=lambda x: x[0])
        top_files = [file for _, file in file_scores[:top_n]]

        return top_files
        """
        relevant_files = []
        for filename, file in self.dataset.items():
            # Check if the file title contains a word similar enough to a keyword
            
            filewords = utils.get_words_from_filename(filename)

            filename_embeddings = utils.get_embeddings(self.tokenizer, self.model, filewords)

            print("filename_embeddings.size() : ", filename_embeddings.size())

            if any(utils.are_similar(keyword_embedding, filename_embedding, cutoff) for keyword_embedding in keywords for filename_embedding in filename_embeddings):
                print("Found a relevant file: ", filename)
                relevant_files.append(file)
        return relevant_files
        """
    
    def retrieveData(self, keywords: List[torch.Tensor], df: pd.DataFrame, top_n: int = 5) -> str:
        similarity_scores = []
        
        # Get all columns that contain strings (object or string dtype)
        string_columns = df.select_dtypes(include=['object', 'string']).columns
        
        # Calculate similarity scores for each cell in the string columns
        for col in string_columns:
            embeddings = df[col].apply(lambda x: utils.get_embeddings(self.tokenizer, self.model, x))
            for idx, cell_embeddings in enumerate(embeddings):
                for keyword in keywords:
                    for embedding in cell_embeddings:
                        score = utils.similarity_score(keyword, embedding)
                        similarity_scores.append((score, idx))
        
        # Sort the similarity scores and keep the top N
        similarity_scores.sort(reverse=True, key=lambda x: x[0])
        top_indices = [idx for _, idx in similarity_scores[:top_n]]
        
        # Return the rows corresponding to the top N similarity scores
        if top_indices:
            return df.iloc[top_indices].to_string(index=False)
        
        return "no data" # Return this message if no data is found