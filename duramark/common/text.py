"""Text processing utilities."""

import regex
import torch
from transformers import BertTokenizer

_tokenizer = None


def _get_tokenizer():
    """Lazy-load the BERT Chinese tokenizer (avoids import-time downloads)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    return _tokenizer


def keep_chinese(text: str) -> str:
    """Strip all non-Chinese characters, keeping only Han characters.

    Args:
        text: Input string.

    Returns:
        String containing only Chinese characters.
    """
    return regex.sub(r'\P{Han}+', '', text)


def extract_text_token(text: str, device: torch.device):
    """Tokenize Chinese text and return token IDs with length.

    Args:
        text: Chinese text string (must be non-empty after keep_chinese).
        device: Target torch device.

    Returns:
        Tuple of (text_token tensor [1, L], text_token_len tensor [1]).
    """
    tokenizer = _get_tokenizer()
    text_token = tokenizer.encode(text, add_special_tokens=False)
    text_token = torch.tensor([text_token], dtype=torch.int32).to(device)
    text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(device)
    return text_token, text_token_len
