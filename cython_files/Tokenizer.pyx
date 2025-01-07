# cython: embedsignature=True, binding=True
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from cython.operator cimport dereference as deref, preincrement as preinc

cdef extern from "<utility>" namespace "std":
    cdef cppclass pair[T, U]:
        T first
        U second

cdef extern from "<unordered_map>" namespace "std":
    cdef cppclass unordered_map[T, U]:
        ctypedef pair[T, U] value_type
        cppclass iterator:
            value_type& operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        iterator begin()
        iterator end()

cdef extern from "<cctype>" namespace "std":
    int tolower(int)
    int isalnum(int)
    int isdigit(int)
    int isspace(int)

cdef class BasicTokenizer:
    cdef unordered_set[string] stop_words
    cdef unordered_map[string, string] contractions
    cdef bint remove_numbers
    cdef bint remove_single_chars
    cdef int min_token_length
    cdef int max_token_length
    
    def __init__(self, bint remove_numbers=False, bint remove_single_chars=True, 
                 int min_token_length=2, int max_token_length=50):
        self.remove_numbers = remove_numbers
        self.remove_single_chars = remove_single_chars
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
        
        # Initialize stop words
        self.stop_words = {
            b"is", b"am", b"are", b"was", b"were",
            b"be", b"been", b"being",
            b"the", b"a", b"an",
            b"in", b"on", b"at", b"to",
            b"and", b"or", b"but", b"if", b"then",
            b"else", b"when", b"up", b"down", b"out",
            b"for", b"of", b"by", b"with", b"about",
            # Add more stop words as needed
        }
        
        # Initialize contractions
        self.contractions = {
            b"'m": b"am",
            b"'s": b"is",
            b"'re": b"are",
            b"'ve": b"have",
            b"'ll": b"will",
            b"n't": b"not",
            b"'d": b"would",
            b"gonna": b"going to",
            b"wanna": b"want to",
            # Add more contractions as needed
        }

    cdef bint is_punctuation(self, char c):
        # Extended punctuation and escape characters
        return (c in b'.,!?;:()[]{}"\'\\'  # basic punctuation
                or c in b'@#$%^&*+=~`|<>/'  # special characters
                or c == b'\n'  # newline
                or c == b'\t'  # tab
                or c == b'\r'  # carriage return
                or c == b'\f'  # form feed
                or c == b'\v'  # vertical tab
                or c == b'\b'  # backspace
                or c == b'\0'  # null character
                or isspace(c)  # any whitespace
                )

    cdef bint is_stop_word(self, const string& token):
        return self.stop_words.find(token) != self.stop_words.end()

    cdef bint is_number(self, const string& token):
        cdef size_t i
        cdef char c
        cdef bint has_digit = False
        
        for i in range(token.length()):
            c = token[i]
            if not isdigit(c) and c != b'.' and c != b'-':
                return False
            if isdigit(c):
                has_digit = True
        return has_digit

    cdef bint is_valid_token(self, const string& token):
        if token.length() < self.min_token_length or token.length() > self.max_token_length:
            return False
        if self.remove_single_chars and token.length() == 1:
            return False
        if self.remove_numbers and self.is_number(token):
            return False
        return True


    cdef string normalize_token(self, string token):
        cdef unordered_map[string, string].iterator it = self.contractions.begin()
        cdef unordered_map[string, string].iterator end = self.contractions.end()
        
        while it != end:
            if token.length() >= deref(it).first.length():
                if token.substr(token.length() - deref(it).first.length()) == deref(it).first:
                    return token.substr(0, token.length() - deref(it).first.length()) + deref(it).second
            preinc(it)  # Using preincrement instead of postincrement
        return token



    def __call__(self, string text):
        cdef vector[string] tokens
        cdef string current_token
        cdef char c
        cdef bint last_was_space = True
        cdef string normalized_token

        for word in range(len(text)):
            c = text[word]
            
            if self.is_punctuation(c):
                if current_token.length() > 0:
                    normalized_token = self.normalize_token(current_token)
                    if self.is_valid_token(normalized_token) and not self.is_stop_word(normalized_token):
                        tokens.push_back(normalized_token)
                    current_token.clear()
                last_was_space = True
            else:
                if not last_was_space or current_token.length() == 0:
                    current_token += <char>tolower(ord(c))
                last_was_space = False

        # Handle the last token
        if current_token.length() > 0:
            normalized_token = self.normalize_token(current_token)
            if self.is_valid_token(normalized_token) and not self.is_stop_word(normalized_token):
                tokens.push_back(normalized_token)
        return tokens