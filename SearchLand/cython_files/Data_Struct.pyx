from libcpp.unordered_map cimport unordered_map
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.math cimport log
from cython.operator import dereference as deref, preincrement as inc
from .Tokenizer import BasicTokenizer
import pickle

cdef struct DocumentInfo:
    int term_frequency
    vector[int] positions


cdef struct Chunk:
    int chunk_id
    string text
    vector[float] embeddings


cdef struct Document:
    int doc_id
    string file_name
    unordered_map[string, DocumentInfo] tokens
    map[int , Chunk] chunks


cdef class InvertedIndex:
    cdef unordered_map[int, Document] arrays  
    cdef unordered_map[string, int] global_term_freq
    cdef object tokenizer
    cdef int chunk_id
    cdef float avg_doc_length
    cdef int total_docs

    def __init__(self):    

        self.avg_doc_length = 0.0
        self.total_docs = 0
        self.arrays = unordered_map[int, Document]()
        self.tokenizer = BasicTokenizer() 
        self.chunk_id = 0

    
    def return_chunk(self, int doc_id, int chunk_id) -> str:
        # C++ string → Python bytes → Python str
        cdef bytes b = self.arrays[doc_id].chunks[chunk_id].text
        return b.decode('utf-8', 'replace')


    def return_close_chunks(self, int doc_id, int chunk_id, int n) -> str:

        if self.arrays.find(doc_id) == self.arrays.end():
            return ""
        if self.arrays[doc_id].chunks.find(chunk_id) == self.arrays[doc_id].chunks.end():
            return ""
        if n <= 0:
            return ""
        if self.arrays[doc_id].chunks.size() == 1:
            return self.arrays[doc_id].chunks[chunk_id].text.decode('utf-8')
        
        cdef vector[int] chunk_ids
        cdef map[int, Chunk].iterator it = self.arrays[doc_id].chunks.begin()
        while it != self.arrays[doc_id].chunks.end():
            chunk_ids.push_back(deref(it).first)
            inc(it)
        
        cdef int target_idx = -1
        for i in range(chunk_ids.size()):
            if chunk_ids[i] == chunk_id:
                target_idx = i
                break
        
        if target_idx == -1:  # This shouldn't happen given our earlier check
            return ""
        
        cdef int start_idx = max(0, target_idx - n//2)
        cdef int end_idx = min(<int>chunk_ids.size(), start_idx + n)
        start_idx = max(0, end_idx - n)  # Adjust start if we hit the right boundary
        
        cdef list chunks = []
        cdef int chunk_id_to_get
        for i in range(start_idx, end_idx):
            chunk_id_to_get = chunk_ids[i]
            b = self.arrays[doc_id].chunks[chunk_id_to_get].text
            chunks.append(b.decode('utf-8','replace'))
        
        return " ".join(chunks)

    def return_doc(self, int doc_id) -> str:
        cdef string text
        cdef map[int, Chunk].iterator it = self.arrays[doc_id].chunks.begin()
        while it != self.arrays[doc_id].chunks.end():
            text += deref(it).second.text
            inc(it)
        return (<bytes>text).decode('utf-8','replace')



    def add_txt(self, str text, int doc_id, str file_name):
        """
        text : a Python unicode string
        """
        # store file_name as UTF-8 in C++ string
        cdef string sfile_name = file_name.encode('utf-8')
        return self.add_to_doc_unicode(text, doc_id, sfile_name)

    def add_txt(self, str text, int doc_id, str file_name, int last_chunk_id = 0):
        """
        text : a Python unicode string
        """
        # store file_name as UTF-8 in C++ string
        cdef string sfile_name = file_name.encode('utf-8')
        if last_chunk_id != 0:
            self.chunk_id = last_chunk_id 

        return self.add_to_doc_unicode(text, doc_id, sfile_name)


    cdef add_to_doc_unicode(self, str utext, int doc_id, string file_name
                            ):
        """
        utext is a Python unicode string.
        We chunk it in *characters*, but store the UTF-8 bytes in each Chunk.
        """
        cdef int chunk_size = 256
        utext = ' '.join(utext.split())
        cdef unordered_map[string, int] word_dict
        cdef unordered_map[string, DocumentInfo] index = unordered_map[string, DocumentInfo]()
        # tokenize on unicode
        py_tokens = self.tokenizer(utext)
        # we will need their UTF-8‐encoded byte-keys for the maps:
        cdef string token
        for i, py_tok in enumerate(py_tokens):
            # map Python str → UTF‑8 bytes → C++ string
            token = py_tok.encode('utf-8')
            word_dict[token] += 1
            self.global_term_freq[token] += 1
            if word_dict[token] == 1:
                doc_info = DocumentInfo()
                doc_info.term_frequency = 1
                doc_info.positions.push_back(i)
                index[token] = doc_info
            else:
                index[token].term_frequency += 1
                index[token].positions.push_back(i)
        cdef Document new_doc
        new_doc.doc_id = doc_id
        new_doc.file_name = file_name
        new_doc.tokens = index

        cdef list texts = []
        cdef list chunks_ids = []
        cdef int j = 0
        cdef int end_pos
        cdef Chunk new_chunk
        cdef string chunk

        chunk_counter = 0

        cdef int min_chunk_size = 120

        # chunking in unicode‐characters space
        text_len = len(utext)
        while j < text_len:
            end_pos = j + chunk_size
            if end_pos > text_len:
                end_pos = text_len
            # we can also look for sentence boundaries in utext
            found_boundary = False
            if end_pos - j >= min_chunk_size:
                k = end_pos
                while k > j + min_chunk_size:
                    ch = utext[k-1]
                    if ch in ('.','!','?'):
                        end_pos = k
                        found_boundary = True
                        break
                    k -= 1

            # now extract the chunk as unicode, then UTF-8‐encode it
            chunk_unicode = utext[j:end_pos]
            chunk_bytes = chunk_unicode.encode('utf-8')
            if end_pos == j:
                end_pos = min(j + 1, text_len)
            if chunk_bytes:
                new_chunk.chunk_id = self.chunk_id
                new_chunk.text = <string>chunk_bytes
                new_doc.chunks[self.chunk_id] = new_chunk
                texts.append(chunk_unicode)
                chunks_ids.append(self.chunk_id)
                self.chunk_id += 1
                chunk_counter += 1

            j = end_pos
        self.arrays[doc_id] = new_doc
        self.total_docs += 1
        self.avg_doc_length = ((self.avg_doc_length * (self.total_docs - 1)) + len(index)) / self.total_docs
        return texts, chunks_ids, doc_id


    def delete_doc(self, int doc_id):
        """
        Deletes a document and its associated data from the index.
        """
        # 1. Check if document exists
        if self.arrays.find(doc_id) == self.arrays.end():
            print(f"Warning: Document with ID {doc_id} not found for deletion.")
            return False # Indicate failure or non-existence

        # 2. Get reference and data before deleting
        cdef Document* doc_to_delete = &self.arrays[doc_id]
        cdef int deleted_doc_unique_token_count = doc_to_delete.tokens.size()

        # 3. Update global term frequencies
        cdef unordered_map[string, DocumentInfo].iterator token_it = doc_to_delete.tokens.begin()
        cdef string current_token
        while token_it != doc_to_delete.tokens.end():
            current_token = deref(token_it).first
            # Decrement global frequency by the term's frequency in the deleted doc
            if self.global_term_freq.count(current_token):
                self.global_term_freq[current_token] -= deref(token_it).second.term_frequency
                # Optional: Remove term from global map if its count drops to 0 or below
                if self.global_term_freq[current_token] <= 0:
                    self.global_term_freq.erase(current_token)
            inc(token_it)

        # 4. Update average document length and total document count
        total_unique_tokens_before = self.avg_doc_length * self.total_docs
        self.total_docs -= 1 # Decrement total docs count

        if self.total_docs > 0:
            # Recalculate average length
            total_unique_tokens_after = total_unique_tokens_before - deleted_doc_unique_token_count
            # Avoid division by zero if total_unique_tokens_after becomes negative (shouldn't happen)
            self.avg_doc_length = max(0.0, total_unique_tokens_after) / self.total_docs
        else:
            # Reset average length if no documents remain
            self.avg_doc_length = 0.0

        # 5. Remove the document entry from the main map
        self.arrays.erase(doc_id)

        print(f"Successfully deleted document with ID {doc_id}.")
        return True # Indicate successful deletion

    def has_doc(self, int doc_id):
        """
        Check if a document with the given ID exists in the index.
        """
        return self.arrays.find(doc_id) != self.arrays.end()


    def add_emmbeddings(self, list[list[float]] embeddings_req, int doc_id):
        for i,embedding  in enumerate(embeddings_req):
            self.arrays[doc_id].chunks[i].embeddings = embedding
            


    def search(self, str sentence, int top_n=5):
        # 1) tokenize on Python unicode
        cdef list py_tokens = self.tokenizer(sentence)
        if not py_tokens:
            return []

        # 2) build C++ vector<string> of UTF-8 tokens
        cdef vector[string] query_tokens
        cdef string tok
        for py_tok in py_tokens:
            tok = py_tok.encode('utf-8')      # Python bytes -> C++ string
            query_tokens.push_back(tok)

        # 3) collect doc_ids from self.arrays
        cdef vector[int] doc_ids
        cdef unordered_map[int, Document].iterator dit = self.arrays.begin()
        while dit != self.arrays.end():
            doc_ids.push_back(deref(dit).first)
            inc(dit)

        # 4) compute document frequency for each term
        cdef unordered_map[string, int] term_doc_freq
        cdef int did
        for tok in query_tokens:
            for did in doc_ids:
                if self.arrays[did].tokens.find(tok) != self.arrays[did].tokens.end():
                    term_doc_freq[tok] += 1

        # 5) BM25 + proximity
        cdef list results = []
        cdef float k1 = 1.5, b = 0.75
        cdef int doc_length
        cdef bint all_present
        cdef dict token_positions
        cdef float bm25, prox, tf, idf, tf_adj, final_score

        for did in doc_ids:
            doc_length = <int>self.arrays[did].tokens.size()
            bm25 = 0.0
            all_present = True
            token_positions = {}

            # score each query token
            for tok in query_tokens:
                if self.arrays[did].tokens.find(tok) != self.arrays[did].tokens.end():
                    # record positions
                    token_positions[tok.decode('utf-8')] = {
                        'positions': list(self.arrays[did].tokens[tok].positions),
                        'term_frequency': self.arrays[did].tokens[tok].term_frequency
                    }
                    # BM25
                    tf   = self.arrays[did].tokens[tok].term_frequency
                    idf  = log((self.total_docs - term_doc_freq[tok] + 0.5) /
                               (term_doc_freq[tok] + 0.5) + 1.0)
                    tf_adj = ((k1 + 1) * tf) / (k1 * (1 - b + b * doc_length / self.avg_doc_length) + tf)
                    bm25 += idf * tf_adj
                else:
                    all_present = False

            if token_positions:
                # proximity bonus
                prox = self._calculate_proximity_score(did, query_tokens)
                final_score = bm25 * (1.0 + prox)
                results.append({
                    'doc_id':          did,
                    'file_name':       self.arrays[did].file_name,
                    'score':           final_score,
                    'bm25_score':      bm25,
                    'proximity_bonus': prox,
                    'positions':       token_positions,
                    'all_terms_present': all_present
                })

        # 6) sort & return top_n
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_n]


    cdef float _calculate_proximity_score(self, int doc_id, vector[string]& query_tokens):
        cdef float proximity_score = 0.0
        cdef vector[vector[int]] term_positions
        cdef int min_distance = 1000000
        cdef int window_size = 10  # Adjustable parameter
        
        # Collect positions
        for token in query_tokens:
            if self.arrays[doc_id].tokens.find(token) != self.arrays[doc_id].tokens.end():
                term_positions.push_back(self.arrays[doc_id].tokens[token].positions)
        
        # Check if we have enough positions to calculate proximity
        if term_positions.size() < 2:
            return 0.0
        
        # Check if any position vector is empty
        for i in range(term_positions.size()):
            if term_positions[i].size() == 0:
                return 0.0
        
        # Use sliding window approach
        cdef vector[int] current_positions = vector[int](term_positions.size(), 0)
        cdef int min_pos, max_pos
        
        while True:
            # Check if any current_positions index is out of bounds
            for i in range(term_positions.size()):
                if current_positions[i] >= term_positions[i].size():
                    return 1.0 / (1.0 + min_distance)
            
            min_pos = term_positions[0][current_positions[0]]
            max_pos = min_pos
            
            for i in range(1, term_positions.size()):
                while (current_positions[i] < term_positions[i].size() and 
                    term_positions[i][current_positions[i]] < min_pos):
                    current_positions[i] += 1
                
                if current_positions[i] >= term_positions[i].size():
                    return 1.0 / (1.0 + min_distance)
                
                pos = term_positions[i][current_positions[i]]
                min_pos = min(min_pos, pos)
                max_pos = max(max_pos, pos)
            
            if max_pos - min_pos < min_distance:
                min_distance = max_pos - min_pos
                
            # Move the pointer of the smallest position forward
            for i in range(term_positions.size()):
                if term_positions[i][current_positions[i]] == min_pos:
                    current_positions[i] += 1
                    if current_positions[i] >= term_positions[i].size():
                        return 1.0 / (1.0 + min_distance)
                    break