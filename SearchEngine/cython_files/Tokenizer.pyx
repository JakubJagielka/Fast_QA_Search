# Tokenizer.pyx
# cython: language_level=3

import unicodedata

cdef set STOP_WORDS = {
    "is","am","are","was","were","be","been","being",
    "the","a","an","in","on","at","to","and","or","but",
    "if","then","else","when","up","down","out","for","of",
    "by","with","about",
}

cdef dict CONTRACTIONS = {
    "'m":"am",
    "'s":"is",
    "'re":"are",
    "'ve":"have",
    "'ll":"will",
    "n't":"not",
    "'d":"would",
    "gonna":"going to",
    "wanna":"want to",
}

cdef class BasicTokenizer:
    cdef int min_token_length, max_token_length
    cdef bint remove_numbers, remove_single_chars

    def __init__(self, bint remove_numbers=False, bint remove_single_chars=True,
        int min_token_length=2, int max_token_length=50):
        self.remove_numbers = remove_numbers
        self.remove_single_chars = remove_single_chars
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length

    cdef bint _is_punctuation(self, str c):
        # anything whose Unicode category starts with 'P' or is whitespace
        return unicodedata.category(c).startswith('P') or c.isspace()

    cdef str _normalize(self, str token):
        for k, v in CONTRACTIONS.items():
            if token.endswith(k):
                return token[:-len(k)] + v
        return token

    cdef bint _is_number(self, str token):
        cdef bint has_digit = False
        for ch in token:
            if ch.isdigit():
                has_digit = True
            elif ch not in ('.','-'):
                return False
        return has_digit

    cdef bint _is_valid(self, str token):
        l = len(token)
        if l < self.min_token_length or l > self.max_token_length:
            return False
        if self.remove_single_chars and l == 1:
            return False
        if self.remove_numbers and self._is_number(token):
            return False
        if token in STOP_WORDS:
            return False
        return True

    def __call__(self, text: str) -> list[str]:
        """
        text: a Python str (Unicode).
        returns: a Python list of Python str tokens.
        """
        cdef list tokens = []
        cdef list curr = []
        for ch in text:
            if self._is_punctuation(ch): 
                if curr:
                    tok = "".join(curr).lower()
                    tok = self._normalize(tok)
                    if self._is_valid(tok):
                        tokens.append(tok)
                    curr.clear()
            else:
                curr.append(ch)
        if curr:
            tok = "".join(curr).lower()
            tok = self._normalize(tok)
            if self._is_valid(tok):
                tokens.append(tok)
        return tokens