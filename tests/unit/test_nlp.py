"""
Unit tests for NLP helper functions.
"""

import numpy as np
import pytest

from ml_core.nlp import (
    TextPreprocessor,
    build_vocabulary,
    clean_text,
    compute_idf,
    compute_tf,
    compute_tfidf,
    cosine_similarity,
    create_attention_mask,
    encode_texts,
    generate_ngrams,
    get_word_frequencies,
    mean_pooling,
    pad_sequences,
    simple_tokenize,
)


class TestCleanText:
    """Tests for clean_text function."""

    def test_lowercase(self):
        result = clean_text("Hello World", lowercase=True)
        assert result == "hello world"

    def test_no_lowercase(self):
        result = clean_text("Hello World", lowercase=False)
        assert result == "Hello World"

    def test_remove_punctuation(self):
        result = clean_text("Hello, world!", remove_punctuation=True)
        assert result == "hello world"

    def test_keep_punctuation(self):
        result = clean_text("Hello, world!", remove_punctuation=False)
        assert "," in result

    def test_remove_numbers(self):
        result = clean_text("Test 123 text", remove_numbers=True)
        assert "123" not in result

    def test_keep_numbers(self):
        result = clean_text("Test 123 text", remove_numbers=False)
        assert "123" in result

    def test_remove_extra_whitespace(self):
        result = clean_text("Hello   world", remove_extra_whitespace=True)
        assert result == "hello world"

    def test_empty_string(self):
        result = clean_text("")
        assert result == ""


class TestSimpleTokenize:
    """Tests for simple_tokenize function."""

    def test_basic_tokenization(self):
        result = simple_tokenize("hello world")
        assert result == ["hello", "world"]

    def test_empty_string(self):
        result = simple_tokenize("")
        assert result == []  # Split on empty string returns empty list

    def test_single_word(self):
        result = simple_tokenize("hello")
        assert result == ["hello"]

    def test_multiple_spaces(self):
        result = simple_tokenize("hello  world")
        # Python's split() handles multiple spaces correctly
        assert result == ["hello", "world"]


class TestBuildVocabulary:
    """Tests for build_vocabulary function."""

    def test_basic_vocabulary(self):
        texts = ["hello world", "world hello"]
        vocab = build_vocabulary(texts)
        assert "hello" in vocab
        assert "world" in vocab

    def test_special_tokens(self):
        texts = ["hello world"]
        vocab = build_vocabulary(texts, special_tokens=["<PAD>", "<UNK>"])
        assert vocab["<PAD>"] == 0
        assert vocab["<UNK>"] == 1

    def test_min_freq(self):
        texts = ["hello hello world"]
        vocab = build_vocabulary(texts, min_freq=2)
        assert "hello" in vocab
        assert "world" not in vocab

    def test_max_vocab_size(self):
        texts = ["a b c d e"]
        vocab = build_vocabulary(texts, max_vocab_size=3)
        # Should include only 3 words (plus special tokens if any)
        assert len(vocab) == 3

    def test_empty_texts(self):
        vocab = build_vocabulary([])
        assert vocab == {}


class TestEncodeTexts:
    """Tests for encode_texts function."""

    def test_basic_encoding(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "hello": 2, "world": 3}
        texts = ["hello world"]
        result = encode_texts(texts, vocab)
        assert result == [[2, 3]]

    def test_unknown_token(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "hello": 2}
        texts = ["hello unknown"]
        result = encode_texts(texts, vocab)
        assert result == [[2, 1]]  # unknown -> 1 (UNK)

    def test_padding(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "hello": 2}
        texts = ["hello"]
        result = encode_texts(texts, vocab, max_length=3, padding=True)
        assert result == [[2, 0, 0]]

    def test_truncation(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "hello": 2, "world": 3}
        texts = ["hello world hello"]
        result = encode_texts(texts, vocab, max_length=2)
        assert result == [[2, 3]]


class TestComputeTF:
    """Tests for compute_tf function."""

    def test_basic_tf(self):
        tf = compute_tf("hello hello world")
        assert tf["hello"] == 2 / 3
        assert tf["world"] == 1 / 3

    def test_empty_document(self):
        tf = compute_tf("")
        assert tf == {}

    def test_single_word(self):
        tf = compute_tf("hello")
        assert tf["hello"] == 1.0


class TestComputeIDF:
    """Tests for compute_idf function."""

    def test_basic_idf(self):
        documents = ["hello world", "hello", "world"]
        idf = compute_idf(documents)
        assert "hello" in idf
        assert "world" in idf
        # hello appears in 2 docs, world appears in 2 docs
        assert idf["hello"] == idf["world"]

    def test_with_vocabulary(self):
        documents = ["hello world", "hello"]
        idf = compute_idf(documents, vocabulary=["hello"])
        assert "hello" in idf
        assert "world" not in idf


class TestComputeTFIDF:
    """Tests for compute_tfidf function."""

    def test_basic_tfidf(self):
        documents = ["hello world", "hello"]
        idf = compute_idf(documents)
        tfidf = compute_tfidf("hello world", idf)
        assert "hello" in tfidf
        assert "world" in tfidf
        assert tfidf["hello"] > 0
        assert tfidf["world"] > 0


class TestGenerateNgrams:
    """Tests for generate_ngrams function."""

    def test_bigrams(self):
        tokens = ["the", "cat", "sat"]
        result = generate_ngrams(tokens, 2)
        assert result == [("the", "cat"), ("cat", "sat")]

    def test_trigrams(self):
        tokens = ["the", "cat", "sat", "down"]
        result = generate_ngrams(tokens, 3)
        assert result == [("the", "cat", "sat"), ("cat", "sat", "down")]

    def test_unigrams(self):
        tokens = ["hello", "world"]
        result = generate_ngrams(tokens, 1)
        assert result == [("hello",), ("world",)]

    def test_invalid_n(self):
        tokens = ["hello"]
        assert generate_ngrams(tokens, 0) == []
        assert generate_ngrams(tokens, 2) == []

    def test_empty_tokens(self):
        result = generate_ngrams([], 2)
        assert result == []


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        vec = np.array([1, 2, 3])
        result = cosine_similarity(vec, vec)
        assert np.isclose(result, 1.0)

    def test_orthogonal_vectors(self):
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        result = cosine_similarity(vec1, vec2)
        assert np.isclose(result, 0.0)

    def test_opposite_vectors(self):
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([-1, -2, -3])
        result = cosine_similarity(vec1, vec2)
        assert np.isclose(result, -1.0)

    def test_zero_vector(self):
        vec1 = np.array([0, 0, 0])
        vec2 = np.array([1, 2, 3])
        result = cosine_similarity(vec1, vec2)
        assert result == 0.0


class TestPadSequences:
    """Tests for pad_sequences function."""

    def test_right_padding(self):
        sequences = [[1, 2], [1, 2, 3]]
        result = pad_sequences(sequences, max_length=4, padding_side="right")
        expected = np.array([[1, 2, 0, 0], [1, 2, 3, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_left_padding(self):
        sequences = [[1, 2], [1, 2, 3]]
        result = pad_sequences(sequences, max_length=4, padding_side="left")
        expected = np.array([[0, 0, 1, 2], [0, 1, 2, 3]])
        np.testing.assert_array_equal(result, expected)

    def test_truncation(self):
        sequences = [[1, 2, 3, 4, 5]]
        result = pad_sequences(sequences, max_length=3)
        expected = np.array([[1, 2, 3]])
        np.testing.assert_array_equal(result, expected)

    def test_custom_padding_value(self):
        sequences = [[1, 2]]
        result = pad_sequences(sequences, max_length=4, padding_value=-1)
        expected = np.array([[1, 2, -1, -1]])
        np.testing.assert_array_equal(result, expected)

    def test_empty_sequences(self):
        result = pad_sequences([])
        assert result.shape == (0, 0)


class TestCreateAttentionMask:
    """Tests for create_attention_mask function."""

    def test_basic_mask(self):
        sequences = [[1, 2, 0, 0], [1, 2, 3, 0]]
        result = create_attention_mask(sequences)
        expected = np.array([[1, 1, 0, 0], [1, 1, 1, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_no_padding(self):
        sequences = [[1, 2, 3]]
        result = create_attention_mask(sequences)
        expected = np.array([[1, 1, 1]])
        np.testing.assert_array_equal(result, expected)


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""

    def test_default_preprocessing(self):
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess("Hello, World!")
        assert result == "hello world"

    def test_stopword_removal(self):
        preprocessor = TextPreprocessor(
            remove_stopwords=True, stopwords=["the", "a", "is"]
        )
        result = preprocessor.preprocess("The cat is a pet")
        assert "the" not in result
        assert "is" not in result
        assert "cat" in result

    def test_custom_transforms(self):
        def replace_numbers(text):
            return text.replace("123", "NUM")

        preprocessor = TextPreprocessor(custom_transforms=[replace_numbers])
        result = preprocessor.preprocess("Test 123")
        # Custom transforms run first, then lowercase is applied
        assert "num" in result.lower()

    def test_batch_preprocessing(self):
        preprocessor = TextPreprocessor()
        texts = ["Hello World", "Test Text"]
        results = preprocessor.preprocess_batch(texts)
        assert results == ["hello world", "test text"]


class TestGetWordFrequencies:
    """Tests for get_word_frequencies function."""

    def test_basic_frequencies(self):
        texts = ["hello world", "hello"]
        result = get_word_frequencies(texts)
        result_dict = dict(result)
        assert result_dict["hello"] == 2
        assert result_dict["world"] == 1

    def test_top_n(self):
        texts = ["a a a b b c"]
        result = get_word_frequencies(texts, top_n=2)
        assert len(result) == 2
        assert result[0][0] == "a"
        assert result[1][0] == "b"

    def test_empty_texts(self):
        result = get_word_frequencies([])
        assert result == []


class TestMeanPooling:
    """Tests for mean_pooling function."""

    def test_basic_pooling(self):
        embeddings = np.array([[[1, 2], [3, 4]]])  # (1, 2, 2)
        result = mean_pooling(embeddings)
        expected = np.array([[2, 3]])  # Mean of [[1,2], [3,4]]
        np.testing.assert_array_almost_equal(result, expected)

    def test_with_attention_mask(self):
        embeddings = np.array([[[1, 2], [3, 4], [5, 6]]])  # (1, 3, 2)
        attention_mask = np.array([[1, 1, 0]])  # Ignore last position
        result = mean_pooling(embeddings, attention_mask)
        expected = np.array([[2, 3]])  # Mean of [[1,2], [3,4]] only
        np.testing.assert_array_almost_equal(result, expected)

    def test_batch_pooling(self):
        embeddings = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])  # (2, 2, 2)
        result = mean_pooling(embeddings)
        expected = np.array([[1.5, 1.5], [3.5, 3.5]])
        np.testing.assert_array_almost_equal(result, expected)
