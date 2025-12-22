"""
NLP helper functions for text processing and analysis.

This module provides utilities for common NLP tasks including:
- Text preprocessing and cleaning
- Tokenization and vocabulary building
- Text vectorization helpers
- Embedding utilities
"""

import re
import string
from collections import Counter
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np


def clean_text(text: str, 
               lowercase: bool = True,
               remove_punctuation: bool = True,
               remove_numbers: bool = False,
               remove_extra_whitespace: bool = True) -> str:
    """
    Clean text by applying various preprocessing steps.
    
    Args:
        text: Input text to clean
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation characters
        remove_numbers: Remove numeric digits
        remove_extra_whitespace: Collapse multiple spaces to single space
        
    Returns:
        Cleaned text string
    """
    if lowercase:
        text = text.lower()
    
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    if remove_extra_whitespace:
        text = ' '.join(text.split())
    
    return text.strip()


def simple_tokenize(text: str) -> List[str]:
    """
    Simple whitespace-based tokenization.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of tokens
    """
    return text.split()


def build_vocabulary(texts: List[str], 
                     min_freq: int = 1,
                     max_vocab_size: Optional[int] = None,
                     special_tokens: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Build vocabulary from a list of texts.
    
    Args:
        texts: List of text documents
        min_freq: Minimum frequency for a word to be included
        max_vocab_size: Maximum vocabulary size (excluding special tokens)
        special_tokens: List of special tokens to add (e.g., ['<PAD>', '<UNK>'])
        
    Returns:
        Dictionary mapping words to indices
    """
    # Count word frequencies
    word_counts = Counter()
    for text in texts:
        words = simple_tokenize(text.lower())
        word_counts.update(words)
    
    # Initialize vocabulary with special tokens
    vocab = {}
    if special_tokens:
        for token in special_tokens:
            vocab[token] = len(vocab)
    
    # Filter by frequency and sort by count
    filtered_words = [word for word, count in word_counts.most_common() 
                      if count >= min_freq]
    
    # Limit vocabulary size
    if max_vocab_size is not None:
        filtered_words = filtered_words[:max_vocab_size]
    
    # Add words to vocabulary
    for word in filtered_words:
        if word not in vocab:
            vocab[word] = len(vocab)
    
    return vocab


def encode_texts(texts: List[str], 
                 vocab: Dict[str, int],
                 max_length: Optional[int] = None,
                 padding: bool = True,
                 unk_token: str = '<UNK>',
                 pad_token: str = '<PAD>') -> List[List[int]]:
    """
    Encode texts as sequences of token indices.
    
    Args:
        texts: List of text strings to encode
        vocab: Vocabulary mapping words to indices
        max_length: Maximum sequence length (truncate or pad)
        padding: Whether to pad sequences to max_length
        unk_token: Token for unknown words
        pad_token: Token for padding
        
    Returns:
        List of encoded sequences
    """
    unk_idx = vocab.get(unk_token, 0)
    pad_idx = vocab.get(pad_token, 0)
    
    encoded = []
    for text in texts:
        words = simple_tokenize(text.lower())
        indices = [vocab.get(word, unk_idx) for word in words]
        
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length]
            elif padding and len(indices) < max_length:
                indices = indices + [pad_idx] * (max_length - len(indices))
        
        encoded.append(indices)
    
    return encoded


def compute_tf(document: str) -> Dict[str, float]:
    """
    Compute term frequency for a document.
    
    Args:
        document: Text document
        
    Returns:
        Dictionary mapping words to their term frequency
    """
    words = simple_tokenize(document.lower())
    word_counts = Counter(words)
    total_words = len(words)
    
    if total_words == 0:
        return {}
    
    return {word: count / total_words for word, count in word_counts.items()}


def compute_idf(documents: List[str], vocabulary: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Compute inverse document frequency for terms.
    
    Args:
        documents: List of text documents
        vocabulary: Optional list of words to compute IDF for
        
    Returns:
        Dictionary mapping words to their IDF values
    """
    n_docs = len(documents)
    
    # Build vocabulary if not provided
    if vocabulary is None:
        vocabulary = set()
        for doc in documents:
            vocabulary.update(simple_tokenize(doc.lower()))
        vocabulary = list(vocabulary)
    
    idf = {}
    for word in vocabulary:
        doc_count = sum(1 for doc in documents if word in doc.lower())
        idf[word] = np.log(n_docs / (doc_count + 1)) + 1
    
    return idf


def compute_tfidf(document: str, idf: Dict[str, float]) -> Dict[str, float]:
    """
    Compute TF-IDF scores for a document.
    
    Args:
        document: Text document
        idf: IDF values from compute_idf
        
    Returns:
        Dictionary mapping words to their TF-IDF scores
    """
    tf = compute_tf(document)
    return {word: tf_val * idf.get(word, 0) for word, tf_val in tf.items()}


def generate_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Generate n-grams from a list of tokens.
    
    Args:
        tokens: List of tokens
        n: Size of n-grams
        
    Returns:
        List of n-gram tuples
    """
    if n < 1 or len(tokens) < n:
        return []
    
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity value between -1 and 1
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def pad_sequences(sequences: List[List[int]], 
                  max_length: Optional[int] = None,
                  padding_value: int = 0,
                  padding_side: str = 'right') -> np.ndarray:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of sequences (lists of integers)
        max_length: Maximum length (default: length of longest sequence)
        padding_value: Value to use for padding
        padding_side: 'right' or 'left' padding
        
    Returns:
        2D numpy array of padded sequences
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences) if sequences else 0
    
    padded = np.full((len(sequences), max_length), padding_value, dtype=np.int64)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        if padding_side == 'right':
            padded[i, :length] = seq[:length]
        else:
            padded[i, -length:] = seq[:length]
    
    return padded


def create_attention_mask(sequences: List[List[int]], 
                          padding_value: int = 0) -> np.ndarray:
    """
    Create attention mask for padded sequences.
    
    Args:
        sequences: Padded sequences
        padding_value: Value used for padding
        
    Returns:
        Attention mask (1 for real tokens, 0 for padding)
    """
    sequences_array = np.array(sequences)
    return (sequences_array != padding_value).astype(np.int64)


class TextPreprocessor:
    """
    Text preprocessing pipeline with configurable steps.
    """
    
    def __init__(self,
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = False,
                 remove_stopwords: bool = False,
                 stopwords: Optional[List[str]] = None,
                 custom_transforms: Optional[List[Callable[[str], str]]] = None):
        """
        Initialize preprocessor.
        
        Args:
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
            remove_numbers: Remove numbers
            remove_stopwords: Remove stopwords
            stopwords: Custom stopwords list
            custom_transforms: List of custom transform functions
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.stopwords = set(stopwords) if stopwords else set()
        self.custom_transforms = custom_transforms or []
    
    def preprocess(self, text: str) -> str:
        """
        Apply all preprocessing steps to text.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Apply custom transforms first
        for transform in self.custom_transforms:
            text = transform(text)
        
        # Standard cleaning
        text = clean_text(
            text,
            lowercase=self.lowercase,
            remove_punctuation=self.remove_punctuation,
            remove_numbers=self.remove_numbers
        )
        
        # Stopword removal
        if self.remove_stopwords and self.stopwords:
            words = text.split()
            words = [w for w in words if w not in self.stopwords]
            text = ' '.join(words)
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Apply preprocessing to a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


def get_word_frequencies(texts: List[str], 
                         top_n: Optional[int] = None) -> List[Tuple[str, int]]:
    """
    Get word frequencies across all texts.
    
    Args:
        texts: List of text documents
        top_n: Return only top N most common words
        
    Returns:
        List of (word, count) tuples sorted by frequency
    """
    word_counts = Counter()
    for text in texts:
        words = simple_tokenize(text.lower())
        word_counts.update(words)
    
    if top_n is not None:
        return word_counts.most_common(top_n)
    return word_counts.most_common()


def mean_pooling(embeddings: np.ndarray, 
                 attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply mean pooling to token embeddings.
    
    Args:
        embeddings: Token embeddings (batch, seq_len, hidden_dim)
        attention_mask: Attention mask (batch, seq_len)
        
    Returns:
        Pooled embeddings (batch, hidden_dim)
    """
    if attention_mask is None:
        return embeddings.mean(axis=1)
    
    # Expand mask for broadcasting
    mask_expanded = np.expand_dims(attention_mask, -1)
    
    # Sum of embeddings weighted by mask
    sum_embeddings = np.sum(embeddings * mask_expanded, axis=1)
    
    # Sum of mask
    sum_mask = np.sum(mask_expanded, axis=1)
    sum_mask = np.clip(sum_mask, 1e-9, None)  # Avoid division by zero
    
    return sum_embeddings / sum_mask
