from libcpp.unordered_map cimport unordered_map
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.math cimport log
from cython_files.Tokenizer import BasicTokenizer
from cython.operator import dereference as deref, preincrement as inc


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

    
    def return_chunk(self, int doc_id, int chunk_id):
        return self.arrays[doc_id].chunks[chunk_id].text.decode('utf-8')


    def return_close_chunks(self, int doc_id, int chunk_id, int n):

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
            chunks.append(self.arrays[doc_id].chunks[chunk_id_to_get].text.decode('utf-8'))
        
        return " ".join(chunks)

    def return_doc(self, int doc_id):
        cdef string text
        cdef map[int, Chunk].iterator it = self.arrays[doc_id].chunks.begin()
        while it != self.arrays[doc_id].chunks.end():
            text += deref(it).second.text
            inc(it)
        return text.decode('utf-8')



    def add_txt(self, bytes text, int doc_id, str file_name):
        cdef string stext = text
        cdef string sfile_name = file_name.encode('utf-8')
        return self.add_to_doc(stext, doc_id, sfile_name)


    cdef add_to_doc(self, string text, int doc_id,string file_name, int chunk_size = 512):
        cdef unordered_map[string, int] word_dict
        cdef unordered_map[string, DocumentInfo] index = unordered_map[string, DocumentInfo]()
        cdef vector[string] tokens = self.tokenizer(text)
        cdef string token
        for i in range(len(tokens)):
            token = tokens[i]
            word_dict[token] += 1
            self.global_term_freq[token] += 1
            if word_dict[token] == 1:  # First occurrence of the token
                doc_info = DocumentInfo()
                doc_info.term_frequency = 1
                doc_info.positions.push_back(i)
                index[token] = doc_info
            else:  # Update existing DocumentInfo
                index[token].term_frequency += 1
                index[token].positions.push_back(i)
        cdef Document new_doc = Document()
        new_doc.doc_id = doc_id
        new_doc.file_name = file_name
        new_doc.tokens = index

        cdef list texts = []
        cdef list chunks_ids = []
        cdef int j = 0
        cdef int end_pos
        cdef Chunk new_chunk
        cdef string chunk

        while j < len(text):
            end_pos = min(j + chunk_size, len(text))  # Ensure we don't go out of bounds
            while end_pos > j and text[end_pos - 1] not in {'.', '!', '?'}:
                end_pos -= 1

            # If no valid ending punctuation is found, use the full chunk size
            if end_pos == j:
                end_pos = min(j + chunk_size, len(text))

            new_chunk.chunk_id = self.chunk_id
            chunks_ids.append(self.chunk_id)
            self.chunk_id += 1
            new_chunk.text = text.substr(j, end_pos - j)
            texts.append(str(new_chunk.text))
            new_doc.chunks[new_chunk.chunk_id] = new_chunk

            j = end_pos  # Move to the next chunk
        
        self.arrays[doc_id] = new_doc
        self.total_docs += 1
        self.avg_doc_length = ((self.avg_doc_length * (self.total_docs - 1)) + len(index)) / self.total_docs
        return texts, chunks_ids, doc_id


    def add_emmbeddings(self, list[list[float]] embeddings_req, int doc_id):
        for i,embedding  in enumerate(embeddings_req):
            self.arrays[doc_id].chunks[i].embeddings = embedding
            


    def search(self, str sentence, int top_n=5):
        
        cdef vector[string] query_tokens = self.tokenizer(sentence.encode('utf-8'))
        cdef list results = []
        cdef unordered_map[string, int] term_doc_freq
        
        # BM25 parameters
        cdef float k1 = 1.5
        cdef float b = 0.75
        
        # Calculate document frequencies
        cdef string token
        cdef int doc_id
        for token in query_tokens:
            for doc_id in range(self.total_docs):
                if self.arrays[doc_id].tokens.find(token) != self.arrays[doc_id].tokens.end():
                    term_doc_freq[token] += 1

        cdef float bm25_score, proximity_score
        cdef int doc_length
        cdef bint all_terms_present
        cdef dict token_positions
        cdef float tf
        cdef float idf
        cdef float tf_adjusted
        
        for doc_id in range(self.total_docs):
            doc_length = len(self.arrays[doc_id].tokens)
            all_terms_present = True
            bm25_score = 0.0
            
            token_positions = {}

            # Check if document contains query terms and calculate BM25
            for token in query_tokens:
                if self.arrays[doc_id].tokens.find(token) != self.arrays[doc_id].tokens.end():
                    # Store positions
                    token_positions[token.decode('utf-8')] = {
                        'positions': list(self.arrays[doc_id].tokens[token].positions),
                        'term_frequency': self.arrays[doc_id].tokens[token].term_frequency
                    }
                    
                    # Calculate BM25 score for this term
                    tf = self.arrays[doc_id].tokens[token].term_frequency
                    idf = log((self.total_docs - term_doc_freq[token] + 0.5) / 
                                    (term_doc_freq[token] + 0.5) + 1.0)
                    tf_adjusted = ((k1 + 1) * tf) / (k1 * (1 - b + b * doc_length / self.avg_doc_length) + tf)
                    bm25_score += idf * tf_adjusted
                else:
                    all_terms_present = False
            
            if len(token_positions) > 0:  # Document contains at least one query term
                # Calculate proximity score
                proximity_score = self._calculate_proximity_score(doc_id, query_tokens)
                
                # Combine scores
                final_score = bm25_score * (1.0 + proximity_score)
                
                results.append({
                    'doc_id': doc_id,
                    'file_name': self.arrays[doc_id].file_name,
                    'score': final_score,
                    'bm25_score': bm25_score,
                    'proximity_bonus': proximity_score,
                    'positions': token_positions,  # Positions of each token in query terms in the document
                    'all_terms_present': all_terms_present # Whether all query terms are present in the document
                })
        # Sort results by score
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

        