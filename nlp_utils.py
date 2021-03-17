### Module containing auxiliary functions and classes for clinical coding NLP using Transformers


## Load text

import os

def load_text_files(file_names, path):
    """
    It loads the text contained in a set of files into a returned list of strings.
    Code adapted from https://stackoverflow.com/questions/33912773/python-read-txt-files-into-a-dataframe
    """
    output = []
    for f in file_names:
        with open(path + f, "r") as file:
            output.append(file.read())
            
    return output


def load_ss_files(file_names, path):
    """
    It loads the start-end pair of each split sentence from a set of files (start + \t + end line-format expected) into a 
    returned dictionary, where keys are file names and values a list of tuples containing the start-end pairs of the 
    split sentences.
    """
    output = dict()
    for f in file_names:
        with open(path + f, "r") as file:
            f_key = f.split('.')[0]
            output[f_key] = []
            for sent in file:
                output[f_key].append(tuple(map(int, sent.strip().split('\t'))))
            
    return output


import numpy as np
import pandas as pd

def process_ner_labels(df_ann):
    df_res = []
    for i in range(df_ann.shape[0]):
        ann_i = df_ann.iloc[i].values
        # Separate discontinuous locations and split each location into start and end offset
        ann_loc_i = ann_i[4]
        for loc in ann_loc_i.split(';'):
            split_loc = loc.split(' ')
            df_res.append(np.concatenate((ann_i[:4], [int(split_loc[0]), int(split_loc[1])])))

    return pd.DataFrame(np.array(df_res), 
                        columns=list(df_ann.columns[:-1]) + ["start", "end"]).drop_duplicates()


from math import ceil

def process_brat_labels(brat_files):
    """
    brat_files: list containing the path of the annotations files in BRAT format (.ann).
    """
    
    df_res = []
    for file in brat_files:
        with open(file) as ann_file:
            doc_name = file.split('/')[-1].split('.')[0]
            i = 0
            for line in ann_file:
                i += 1
                line_split = line.strip().split('\t')
                assert len(line_split) == 3
                if i % 2 > 0:
                    # BRAT annotation
                    assert line_split[0] == "T" + str(ceil(i/2))
                    text_ref = line_split[2]
                    location = ' '.join(line_split[1].split(' ')[1:]).split(';')
                else:
                    # Code assignment
                    assert line_split[0] == "#" + str(ceil(i/2))
                    code = line_split[2]
                    for loc in location:
                        split_loc = loc.split(' ')
                        df_res.append([doc_name, code, text_ref, int(split_loc[0]), int(split_loc[1])])

    return pd.DataFrame(df_res, 
columns=["doc_id", "code", "text_ref", "start", "end"])
    
        

## Creation of a corpus of annotated fragments. It should be noted that the code in this section could be improved by using 
# Object-Oriented Programming, namely the Template Dessign Pattern (instead of simply creating different functions 
# for different transformer-based models).

# BERT model

# Our aim is to use the same tokenizer the Keras BERT library applies.
# For this reason, the next code is adapted from: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py

import unicodedata

def is_punctuation(ch):
    code = ord(ch)
    return 33 <= code <= 47 or \
        58 <= code <= 64 or \
        91 <= code <= 96 or \
        123 <= code <= 126 or \
        unicodedata.category(ch).startswith('P')

def is_cjk_character(ch):
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F

def is_space(ch):
    return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
        unicodedata.category(ch) == 'Zs'

def is_control(ch):
    return unicodedata.category(ch) in ('Cc', 'Cf')


def word_piece_tokenize(word, word_pos, token_dict):
    """
    word_pos: list containing the start position of each of the characters forming the word
    Code taken from: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py#L121
    """
    
    if word in token_dict:
        return [word], [(word_pos[0], word_pos[-1]+1)]
    sub_tokens, start_end_sub = [], []
    start, stop = 0, 0
    while start < len(word):
        stop = len(word)
        while stop > start:
            sub = word[start:stop]
            if start > 0:
                sub = '##' + sub
            if sub in token_dict:
                break
            stop -= 1
        if start == stop:
            # When len(sub) = 1 and sub is not in token_dict (unk sub-token)
            stop += 1
        sub_tokens.append(sub)
        # Following brat standoff format (https://brat.nlplab.org/standoff.html), end position
        # is the first character position after the considered sub-token
        start_end_sub.append((word_pos[start], word_pos[stop-1]+1))
        start = stop
    return sub_tokens, start_end_sub


def start_end_tokenize_bert(text, token_dict, start_i=0, cased=True):
    """
    Our aim is to produce both a list of sub-tokens and a list of tuples containing the start and
    end char positions of each sub-token.
    
    start_i: the start position of the first character in the text.
    
    Code adapted from: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py#L101
    """
    
    if not cased:
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        text = text.lower()
    spaced = ''
    # Store the start positions of each considered character (ch) in start_arr, 
    # such that sum([len(word) for word in spaced.strip().split()]) = len(start_arr)
    start_arr = [] 
    for ch in text:
        if is_punctuation(ch) or is_cjk_character(ch):
            spaced += ' ' + ch + ' '
            start_arr.append(start_i)
        elif is_space(ch):
            spaced += ' '
        elif not(ord(ch) == 0 or ord(ch) == 0xfffd or is_control(ch)):
            spaced += ch
            start_arr.append(start_i)
        # If it is a control char we skip it but take its offset into account
        start_i += 1
    
    tokens, start_end_arr = [], []
    i = 0
    for word in spaced.strip().split():
        j = i + len(word)
        sub_tokens, start_end_sub = word_piece_tokenize(word, start_arr[i:j], token_dict)
        tokens += sub_tokens
        start_end_arr += start_end_sub
        i = j
        
    return tokens, start_end_arr


def ss_start_end_tokenize_bert(ss_start_end, max_seq_len, text, token_dict, cased=True):
    """
    ss_start_end: list of tuples, where each tuple contains the start-end character positions pair of 
                  the split sentences from the input document text.
    text: document text.
    
    return: two lists of lists, the first for the sub-tokens from the re-split sentences and the second for the 
            start-end char positions pairs of the sub-tokens from the re-split sentences.
    """
    out_sub_token, out_start_end = [], []
    for ss_start, ss_end in ss_start_end:
        sub_token, start_end = start_end_tokenize_bert(text=text[ss_start:ss_end], token_dict=token_dict, 
                                                  start_i=ss_start, cased=cased)
        assert len(sub_token) == len(start_end)
        # Re-split large sub-tokens sentences
        for i in range(0, len(sub_token), max_seq_len):
            out_sub_token.append(sub_token[i:i+max_seq_len])
            out_start_end.append(start_end[i:i+max_seq_len])
    
    return out_sub_token, out_start_end


def ss_fragment_greedy(ss_token, ss_start_end, max_seq_len):
    """
    Implementation of the multiple-sentence fine-tuning approach developed in http://ceur-ws.org/Vol-2664/cantemist_paper15.pdf,
    which consists in generating text fragments containing the maximum number of adjacent split sentences, such that the length of 
    each fragment is <= max_seq_len.
    
    ss_token and ss_start_end: list of lists of sub-tokens and sub-tokens start-end pairs from the re-split sentences.
    
    return: two lists of lists representing the obtained sub-tokens fragments, comprising a sequence of multiple contiguous sentences.
    """
    frag_token, frag_start_end = [[]], [[]]
    i = 0
    while i < len(ss_token):
        assert len(ss_token[i]) <= max_seq_len
        if len(frag_token[-1]) + len(ss_token[i]) > max_seq_len:
            # Fragment is full, so create a new empty fragment
            frag_token.append([])
            frag_start_end.append([])
            
        frag_token[-1].extend(ss_token[i])
        frag_start_end[-1].extend(ss_start_end[i])
        i += 1
          
    return frag_token, frag_start_end
        

def convert_token_to_id_segment(token_list, tokenizer, seq_len):
    """
    Given a list of sub-tokens representing a single text fragment, and a tokenizer, it returns their corresponding lists of 
    indices and segments. Padding is added as appropriate.
    
    Code adapted from https://github.com/CyberZHG/keras-bert/tree/master/keras_bert/tokenizer.py#L72
    """
    
    # Add [CLS] and [SEP] tokens (second_len = 0)
    tokens, first_len, second_len = tokenizer._pack(token_list, None)
    # Generate idices and segments
    token_ids = tokenizer._convert_tokens_to_ids(tokens)
    segment_ids = [0] * first_len + [1] * second_len
    
    # Padding
    pad_len = seq_len - first_len - second_len
    token_ids += [tokenizer._pad_index] * pad_len
    segment_ids += [0] * pad_len

    return token_ids, segment_ids


from tqdm import tqdm
from copy import deepcopy

def ss_create_frag_input_data_bert(df_text, text_col, df_ann, doc_list, ss_dict, tokenizer, lab_encoder, seq_len, 
                                          greedy=True, cased=True):
    """
    This function generates the data needed to fine-tune BERT on a multi-label text fragment classification
    task. It is an implementation of 3 distinct fragment-based classification approaches: the text-stream strategy developed
    in http://ceur-ws.org/Vol-2696/paper_101.pdf, the multiple-sentence approach developed in 
    http://ceur-ws.org/Vol-2664/cantemist_paper15.pdf and the single-sentence approach described in this study.
    
    df_text: DataFrame containing the documents IDs ("doc_id" column expected) and the text from the documents.
    text_col: name of the column of df_text DataFrame that contains the text from the documents.
    df_ann: DataFrame containing the NER-N annotations of the documents, in the same format as the DataFrame 
            returned by process_ner_labels or process_brat_labels function.
    doc_list: list containing the documents IDs to be considered. df_text, df_ann and ss_dict are expected to
              contain all documents present in doc_list.
    ss_dict: dict where keys are documents IDs and each value is a list of tuples containing the start-end char positions 
             pairs of the split sentences in each document. It uses the same format as the dict returned by the 
             load_ss_files function. If None, the function implements the text-stream fragment-based classification approach.
    tokenizer: keras_bert Tokenizer instance.
    lab_encoder: MultiLabelBinarizer instance already fit.
    seq_len: BERT maximum input sequence size.
    greedy: boolean parameter indicating the strategy followed to generate the text fragments. 
            If True, the multiple-sentence approach is followed, which consists in generating fragments containing the maximum 
            number of adjacent split sentences, such that the length of each fragment is <= seq_len-2.
            If False, the single-sentence strategy is implemented.
    cased: boolean parameter indicating whether casing is preserved during text tokenization.
    
    returns: indices: np.array of shape total_n_frag x seq_len, containing the BERT indices of each generated 
                      sub-tokens fragment.
             segments: np.array of shape total_n_frag x seq_len, containing the BERT segments of each generated 
                       sub-tokens fragment.
             labels: np.array of shape total_n_frag x n_labels, containing the one-hot vector of the labels 
                     with which each generated fragment is annotated.
             n_fragments: np.array of shape n_doc, containing the number of fragments generated for each document.
    """
    indices, segments, labels, n_fragments, start_end_offsets = [], [], [], [], []
    for doc in tqdm(doc_list):
        # Extract doc annotation
        doc_ann = df_ann[df_ann["doc_id"] == doc]
        # Extract doc text
        doc_text = df_text[df_text["doc_id"] == doc][text_col].values[0]
        # Generate fragments
        if ss_dict is not None:
            # Perform sentence split (SS) on doc text
            doc_ss = ss_dict[doc] # SS start-end pairs of the doc text
            doc_ss_token, doc_ss_start_end = ss_start_end_tokenize_bert(ss_start_end=doc_ss, 
                                    max_seq_len=seq_len-2, text=doc_text, 
                                    token_dict=tokenizer._token_dict, cased=cased)
            assert len(doc_ss_token) == len(doc_ss_start_end)
            if greedy:
                # Split the list of sub-tokens sentences into fragments using multiple-sentence strategy
                frag_token, frag_start_end = ss_fragment_greedy(ss_token=doc_ss_token, 
                               ss_start_end=doc_ss_start_end, max_seq_len=seq_len-2)
            else: 
                frag_token = deepcopy(doc_ss_token)
                frag_start_end = deepcopy(doc_ss_start_end)
        else:
            # Generate text fragments using text-stream strategy (without considering SS)
            doc_token, doc_start_end = start_end_tokenize_bert(text=doc_text, token_dict=tokenizer._token_dict, 
                                                          cased=cased)
            assert len(doc_token) == len(doc_start_end)
            frag_token, frag_start_end = [], []
            for i in range(0, len(doc_token), seq_len-2):
                frag_token.append(doc_token[i:i+seq_len-2])
                frag_start_end.append(doc_start_end[i:i+seq_len-2])
                
        assert len(frag_token) == len(frag_start_end)
        # Store the start-end char positions of all the fragments
        start_end_offsets.extend(frag_start_end)
        # Store the number of fragments of each doc text
        n_fragments.append(len(frag_token))
        
        # Assign to each fragment the labels (codes) from the NER-N annotations exclusively occurring inside 
        # the fragment
        for f_token, f_start_end in zip(frag_token, frag_start_end):
            # fragment length is assumed to be <= SEQ_LEN-2
            assert len(f_token) == len(f_start_end) <= seq_len-2
            # Indices & Segments
            frag_id, frag_seg = convert_token_to_id_segment(f_token, tokenizer, seq_len)
            indices.append(frag_id)
            segments.append(frag_seg)
            # Labels
            frag_labels = []
            # start-end char positions of the whole fragment, i.e. the start position of the first
            # sub-token and the end position of the last sub-token
            frag_start, frag_end = f_start_end[0][0], f_start_end[-1][1]
            for j in range(doc_ann.shape[0]):
                doc_ann_cur = doc_ann.iloc[j] # current annotation
                # Add the annotations whose text references are contained within the fragment
                if doc_ann_cur['start'] < frag_end and doc_ann_cur['end'] > frag_start:
                    frag_labels.append(doc_ann_cur['code'])
            labels.append(frag_labels)
    
    # start_end_offsets is returned for further sanity checking purposes only
    return np.array(indices), np.array(segments), lab_encoder.transform(labels), np.array(n_fragments), start_end_offsets



# XLM-R model

def convert_char_to_bytes_offset(text_str, encoding='utf-8', err_arg='strict'):
    """
    This functions returns a list of pairs in which each pair represents the bytes start-end offset 
    of each character in the input text, according to a character encoding, e.g. UTF-8.
    
    The end offset corresponds to the first byte of the next char.
    
    text_str: string containing the text to be converted into bytes offsets.
    """
    
    start_end_bytes = []
    start_b = 0
    for ch in text_str:
        ch_len_b = len(ch.encode(encoding, err_arg))
        start_end_bytes.append((start_b, start_b + ch_len_b))
        start_b += ch_len_b
    
    return start_end_bytes


def start_end_tokenize_xlmr(text, sp_model, sp_pb2, start_byte=0):
    """
    Our aim is to produce both a list of SentencePiece (SP) sub-tokens IDs and a list of tuples containing the bytes start-end offset
    of each sub-token obtained from a given text.
    
    start_byte: the byte starting position of the first character in the text.
    """
    
    sp_pb2.ParseFromString(sp_model.encode_as_serialized_proto(text))
    
    tokens_id, start_end_arr = [], []
    for p in sp_pb2.pieces:
        tokens_id.append(p.id)
        start_end_arr.append((p.begin + start_byte, p.end + start_byte))
    
    return tokens_id, start_end_arr



def ss_start_end_tokenize_xlmr(ss_start_end, max_seq_len, text, text_char_bytes, sp_model, sp_pb2):
    """
    ss_start_end: list of tuples, where each tuple contains the start-end character positions pair of 
                  the split sentences from the input document text.
    text: document text
    
    return: two lists of lists, the first for the SP sub-tokens IDs from the re-split sentences and the second for the 
            sub-tokens bytes start-end offset pairs.
    """
    out_sub_token_id, out_start_end = [], []
    for ss_start, ss_end in ss_start_end:
        sub_token_id, start_end = start_end_tokenize_xlmr(text=text[ss_start:ss_end], sp_model=sp_model, sp_pb2=sp_pb2,
                                                     start_byte=text_char_bytes[ss_start][0])
        assert len(sub_token_id) == len(start_end)
        # Re-split large sub-tokens sentences
        for i in range(0, len(sub_token_id), max_seq_len):
            out_sub_token_id.append(sub_token_id[i:i+max_seq_len])
            out_start_end.append(start_end[i:i+max_seq_len])
    
    return out_sub_token_id, out_start_end


def convert_token_to_id_attention(token_list, tokenizer, seq_len):
    """
    Given a list of SP sub-tokens IDs representing a single text fragment, and a tokenizer, it returns their correponding lists of 
    indices and attention masks. Padding is added as appropriate.
    """
    
    # Convert SP IDs to correponding model vocabulary IDs
    for i in range(len(token_list)):
        token_list[i] = token_list[i] + tokenizer.fairseq_offset if token_list[i] else tokenizer.unk_token_id
    
    # Add <s> (equiv. to [CLS]) and </s> (equiv. to [SEP]) tokens
    token_ids = [tokenizer.cls_token_id] + token_list + [tokenizer.sep_token_id]
    
    # Generate attention mask
    first_len = len(token_ids)
    attention_mask = [1] * first_len
    
    # Padding
    pad_len = seq_len - first_len
    token_ids += [tokenizer.pad_token_id] * pad_len
    attention_mask += [0] * pad_len

    return token_ids, attention_mask


def ss_create_frag_input_data_xlmr(df_text, text_col, df_ann, doc_list, ss_dict, tokenizer, sp_pb2, lab_encoder, seq_len, 
                                          encoding='utf-8', err_arg='strict'):
    """
    This function generates the data needed to fine-tune XLM-R on a multi-label text fragment classification
    task. Concretely, it is an implementation of the single-sentence approach described in this study.
    
    This function is very similar to ss_create_frag_input_data_bert. In future versions, the Template Dessign Pattern
    will be applied in order to merge both functions.
    
    tokenizer: HuggingFace XLMRobertaTokenizer instance.
    sp_pb2: SentencePieceText instance.
    
    See ss_create_frag_input_data_bert for a description of other input parameters.
    
    returns: indices: np.array of shape total_n_frag x seq_len, containing the XLM-R indices of each generated 
                      sub-tokens fragment.
             attention_mask: np.array of shape total_n_frag x seq_len, containing the XLM-R attention mask of 
                       each generated sub-tokens fragment.
             labels: np.array of shape total_n_frag x n_labels, containing the one-hot vector of the labels 
                     with which each generated fragment is annotated.
             n_fragments: np.array of shape n_doc, containing the number of fragments generated for each document.
    """
    indices, attention_mask, labels, n_fragments, start_end_offsets = [], [], [], [], []
    for doc in tqdm(doc_list):
        # Extract doc annotation
        doc_ann = df_ann[df_ann["doc_id"] == doc]
        # Extract doc text
        doc_text = df_text[df_text["doc_id"] == doc][text_col].values[0]
        # Extract bytes start-end offset of each character in the text
        ch_start_end_bytes = convert_char_to_bytes_offset(doc_text, encoding, err_arg)
        
        # Generate fragments
        # Perform SS on doc text
        doc_ss = ss_dict[doc] # SS start-end pairs of the doc text
        doc_ss_token_id, doc_ss_start_end = ss_start_end_tokenize_xlmr(ss_start_end=doc_ss, 
                                    max_seq_len=seq_len-2, text=doc_text, text_char_bytes=ch_start_end_bytes,
                                    sp_model=tokenizer.sp_model, sp_pb2=sp_pb2)
        assert len(doc_ss_token_id) == len(doc_ss_start_end)
        frag_token = deepcopy(doc_ss_token_id)
        frag_start_end = deepcopy(doc_ss_start_end)
                
        assert len(frag_token) == len(frag_start_end)
        # Store the start-end bytes positions of all the fragments
        start_end_offsets.extend(frag_start_end)
        # Store the number of fragments of each doc text
        n_fragments.append(len(frag_token))
        
        ## Assign to each fragment the labels (codes) from the NER-N annotations exclusively occurring inside 
        # the fragment
        for f_token, f_start_end in zip(frag_token, frag_start_end):
            # fragment length is assumed to be <= SEQ_LEN-2
            assert len(f_token) == len(f_start_end) <= seq_len-2
            # Indices & Attention-Mask
            frag_id, frag_att = convert_token_to_id_attention(f_token, tokenizer, seq_len)
            indices.append(frag_id)
            attention_mask.append(frag_att)
            # Labels
            frag_labels = []
            # bytes start-end positions of the whole fragment, i.e. the start position of the first
            # sub-token and the end position of the last sub-token
            frag_start, frag_end = f_start_end[0][0], f_start_end[-1][1]
            for j in range(doc_ann.shape[0]):
                doc_ann_cur = doc_ann.iloc[j] # current annotation
                # Add the annotations whose text references are contained within the fragment
                if ch_start_end_bytes[doc_ann_cur['start']][0] < frag_end and ch_start_end_bytes[doc_ann_cur['end'] - 1][1] > frag_start:
                    frag_labels.append(doc_ann_cur['code'])
            labels.append(frag_labels)
    
    # start_end_offsets is returned for further sanity checking purposes only
    return np.array(indices), np.array(attention_mask), lab_encoder.transform(labels), np.array(n_fragments), start_end_offsets



## CodiEsp-D abstracts annotations

# BERT model

def create_frag_input_data_bert(df_text, text_col, df_label, doc_list, tokenizer, lab_encoder, seq_len):
    indices, segments, labels, fragments = [], [], [], []
    for doc in tqdm(doc_list):
        # Extract labels
        doc_labels = list(df_label[df_label["doc_id"] == doc]["code"])
        # Tokenize doc text into a list of tokens
        doc_token = tokenizer._tokenize(df_text[df_text["doc_id"] == doc][text_col].values[0])
        # Split the list of tokens (doc) into fragments, and convert token fragments into indices & segments
        n_frag = 0
        for i in range(0, len(doc_token), seq_len-2):
            n_frag += 1
            frag_token = doc_token[i:i+seq_len-2]
            frag_id, frag_seg = convert_token_to_id_segment(frag_token, tokenizer, seq_len)
            indices.append(frag_id)
            segments.append(frag_seg)
            labels.append(doc_labels)
        
        # Store the number of fragments of each doc text
        fragments.append(n_frag)
        
    return np.array(indices), np.array(segments), lab_encoder.transform(labels), np.array(fragments)

# As all abstracts have one single fragment, the next method is compatible with NER method (previous function) to generate data

def create_frag_input_data(df_text, text_col, df_label, doc_list, tokenizer, sp_pb2, lab_encoder, seq_len):
    indices, attention_mask, labels, fragments = [], [], [], []
    for doc in tqdm(doc_list):
        # Extract labels
        doc_labels = list(df_label[df_label["doc_id"] == doc]["code"])
        # Tokenize doc text into a list of tokens
        doc_token, _ = start_end_tokenize_xlmr(text=df_text[df_text["doc_id"] == doc][text_col].values[0], 
                                         sp_model=tokenizer.sp_model, sp_pb2=sp_pb2)
        # Split the list of tokens (doc) into fragments, and convert token fragments into indices & segments
        n_frag = 0
        for i in range(0, len(doc_token), seq_len-2):
            n_frag += 1
            frag_token = doc_token[i:i+seq_len-2]
            frag_id, frag_att = convert_token_to_id_attention(frag_token, tokenizer, seq_len)
            indices.append(frag_id)
            attention_mask.append(frag_att)
            labels.append(doc_labels)
        
        # Store the number of fragments of each doc text
        fragments.append(n_frag)
        
    return np.array(indices), np.array(attention_mask), lab_encoder.transform(labels), np.array(fragments)



## Models evaluation

def max_fragment(y_frag_pred, n_fragments):
    """
    Convert fragment-level to document-level predictions, using maximum porbability criterion.
    """
    y_pred = []
    i_frag = 0
    for n_frag in n_fragments:
        y_pred.append(y_frag_pred[i_frag:i_frag+n_frag].max(axis=0))
        i_frag += n_frag
    return np.array(y_pred)


def max_fragment_prediction(y_frag_pred, n_fragments, label_encoder_classes, doc_list):
    """
    Convert fragment-level to document-level predictions in CodiEsp submission format.
    """
    return prob_codiesp_prediction_format(max_fragment(y_frag_pred, n_fragments), label_encoder_classes, doc_list)


def prob_codiesp_prediction_format(y_pred, label_encoder_classes, doc_list):
    """
    Given a matrix of predicted probabilities (m_docs x n_codes), for each document, this procedure stores all the
    codes sorted according to their probability values in descending order. Finally, predictions are saved in a dataframe
    defined following CodiEsp submission format (see https://temu.bsc.es/codiesp/index.php/2020/02/06/submission/).
    """
    
    # Sanity check
    assert y_pred.shape[0] == len(doc_list)
    
    pred_doc, pred_code, pred_rank = [], [], []
    for i in range(y_pred.shape[0]):
        pred = y_pred[i]
        # Codes are sorted according to their probability values in descending order
        codes_sort = [label_encoder_classes[j] for j in np.argsort(pred)[::-1]]
        pred_code += codes_sort
        pred_doc += [doc_list[i]]*len(codes_sort)
        # For compatibility with format_predictions function
        pred_rank += list(range(1, len(codes_sort)+1))
            
    # Save predictions in CodiEsp submission format
    return pd.DataFrame({"doc_id": pred_doc, "code": pred_code, "rank": pred_rank})


def thr_codiesp_prediction_format(y_pred, label_encoder_classes, doc_list, thr=0.1):
    """
    Given a matrix of predicted probabilities (m_docs x n_codes), for each document, this procedure stores all the
    codes having a probability value higher than a given threshold, sorted according to their probability value.
    """
    
    # Sanity check
    assert y_pred.shape[0] == len(doc_list)
    
    pred_doc, pred_code, pred_rank = [], [], []
    for i in range(y_pred.shape[0]):
        pred = y_pred[i]
        # Codes are sorted according to their probability values in descending order, and then prob threshold is applied
        codes_sort = [label_encoder_classes[j] for j in np.argsort(pred)[::-1]]
        
        pred_code += codes_sort
        pred_doc += [doc_list[i]]*len(codes_sort)
        # For compatibility with format_predictions function
        pred_rank += list(range(1, len(codes_sort)+1))
            
    # Save predictions in CodiEsp submission format
    return pd.DataFrame({"doc_id": pred_doc, "code": pred_code, "rank": pred_rank})


# MAP score
# Code adapted from: https://github.com/TeMU-BSC/CodiEsp-Evaluation-Script/blob/master/codiespD_P_evaluation.py

from trectools import TrecQrel, TrecRun, TrecEval

    
def format_predictions(pred, output_path, valid_codes, 
                       system_name = 'xx', pred_names = ['query','docid', 'rank']):
    '''
    DESCRIPTION: Add extra columns to Predictions table to match 
    trectools library standards.
        
    INPUT: 
        pred: pd.DataFrame
                Predictions.
        output_path: str
            route to TSV where intermediate file is stored
        valid_codes: set
            set of valid codes of this subtask

    OUTPUT: 
        stores TSV files with columns  with columns ['query', "q0", 'docid', 'rank', 'score', 'system']
    
    Note: Dataframe headers chosen to match library standards.
          More informative INPUT headers would be: 
          ["clinical case","code"]

    https://github.com/joaopalotti/trectools#file-formats
    '''
    # Rename columns
    pred.columns = pred_names
    
    # Not needed to: Check if predictions are empty, as all codes sorted by prob, prob-thr etc., are returned
    
    # Add columns needed for the library to properly import the dataframe
    pred['q0'] = 'Q0'
    pred['score'] = float(10) 
    pred['system'] = system_name 
    
    # Reorder and rename columns
    pred = pred[['query', "q0", 'docid', 'rank', 'score', 'system']]
    
    # Not needed to Lowercase codes
    
    # Not needed to: Remove codes predicted twice in the same clinical case
    
    # Not needed to: Remove codes predicted but not in list of valid codes
    
    # Write dataframe to Run file
    pred.to_csv(output_path, index=False, header=None, sep = '\t')


def compute_map(valid_codes, pred, gs_out_path=None):
    """
    Custom function to compute MAP evaluation metric. 
    Code adapted from https://github.com/TeMU-BSC/CodiEsp-Evaluation-Script/blob/master/codiespD_P_evaluation.py
    """
    
    # Input args default values
    if gs_out_path is None: gs_out_path = './intermediate_gs_file.txt' 
    
    pred_out_path = './intermediate_predictions_file.txt'
    ###### 2. Format predictions as TrecRun format: ######
    format_predictions(pred, pred_out_path, valid_codes)
    
    
    ###### 3. Calculate MAP ######
    # Load GS from qrel file
    qrels = TrecQrel(gs_out_path)

    # Load pred from run file
    run = TrecRun(pred_out_path)

    # Calculate MAP
    te = TrecEval(run, qrels)
    MAP = te.get_map(trec_eval=False) # With this option False, rank order is taken from the given document order
    
    ###### 4. Return results ######
    return MAP


# Code copied from: https://github.com/TeMU-BSC/CodiEsp-Evaluation-Script/blob/master/codiespD_P_evaluation.py

def format_gs(filepath, output_path=None, gs_names = ['qid', 'docno']):
    '''
    DESCRIPTION: Load Gold Standard table.
    
    INPUT: 
        filepath: str
            route to TSV file with Gold Standard.
        output_path: str
            route to TSV where intermediate file is stored
    
    OUTPUT: 
        stores TSV files with columns ["query", "q0", "docid", "rel"].
    
    Note: Dataframe headers chosen to match library standards. 
          More informative headers for the INPUT would be: 
          ["clinical case","label","code","relevance"]
    
    # https://github.com/joaopalotti/trectools#file-formats
    '''
    # Input args default values
    if output_path is None: output_path = './intermediate_gs_file.txt' 
    
    # Check GS format:
    check = pd.read_csv(filepath, sep='\t', header = None, nrows=1)
    if check.shape[1] != 2:
        raise ImportError('The GS file does not have 2 columns. Then, it was not imported')
    
    # Import GS
    gs = pd.read_csv(filepath, sep='\t', header = None, names = gs_names)  
        
    # Preprocessing
    gs["q0"] = str(0) # column with all zeros (q0) # Columnn needed for the library to properly import the dataframe
    gs["rel"] = str(1) # column indicating the relevance of the code (in GS, all codes are relevant)
    gs.docno = gs.docno.str.lower() # Lowercase codes
    gs = gs[['qid', 'q0', 'docno', 'rel']]
    
    # Remove codes predicted twice in the same clinical case 
    # (they are present in GS because one code may have several references)
    gs = gs.drop_duplicates(subset=['qid','docno'],  
                            keep='first')  # Keep first of the predictions

    # Write dataframe to Qrel file
    gs.to_csv(output_path, index=False, header=None, sep=' ')

    

# P, R, F1-score
# Code copied from: https://github.com/TeMU-BSC/codiesp-evaluation-script/blob/master/comp_f1_diag_proc.py

import warnings

def read_gs(gs_path):
    gs_data = pd.read_csv(gs_path, sep="\t", names=['clinical_case', 'code'],
                          dtype={'clinical_case': object, 'code':object})
    gs_data.code = gs_data.code.str.lower()
    return gs_data

def read_run(pred_path, valid_codes):
    run_data = pd.read_csv(pred_path, sep="\t", names=['clinical_case', 'code'],
                          dtype={'clinical_case': object, 'code':object})
    run_data.code = run_data.code.str.lower()

    run_data = run_data[run_data['code'].isin(valid_codes)]
    if (run_data.shape[0] == 0):
        warnings.warn('None of the predicted codes are considered valid codes')
    return run_data


def calculate_metrics(df_gs, df_pred):
    Pred_Pos_per_cc = df_pred.drop_duplicates(subset=['clinical_case', 
                                                  "code"]).groupby("clinical_case")["code"].count()
    Pred_Pos = df_pred.drop_duplicates(subset=['clinical_case', "code"]).shape[0]
    
    # Gold Standard Positives:
    GS_Pos_per_cc = df_gs.drop_duplicates(subset=['clinical_case', 
                                               "code"]).groupby("clinical_case")["code"].count()
    GS_Pos = df_gs.drop_duplicates(subset=['clinical_case', "code"]).shape[0]
    cc = set(df_gs.clinical_case.tolist())
    TP_per_cc = pd.Series(dtype=float)
    for c in cc:
        pred = set(df_pred.loc[df_pred['clinical_case']==c,'code'].values)
        gs = set(df_gs.loc[df_gs['clinical_case']==c,'code'].values)
        TP_per_cc[c] = len(pred.intersection(gs))
        
    TP = sum(TP_per_cc.values)
        
    
    # Calculate Final Metrics:
    P_per_cc =  TP_per_cc / Pred_Pos_per_cc
    P = TP / Pred_Pos
    R_per_cc = TP_per_cc / GS_Pos_per_cc
    R = TP / GS_Pos
    F1_per_cc = (2 * P_per_cc * R_per_cc) / (P_per_cc + R_per_cc)
    if (P+R) == 0:
        F1 = 0
        warnings.warn('Global F1 score automatically set to zero to avoid division by zero')
        return P_per_cc, P, R_per_cc, R, F1_per_cc, F1
    F1 = (2 * P * R) / (P + R)
    
    return P_per_cc, P, R_per_cc, R, F1_per_cc, F1


def compute_p_r_f1(gs_path, pred_path, codes_path):
    ###### 0. Load valid codes lists: ######
    valid_codes = set(pd.read_csv(codes_path, sep='\t', header=None, 
                                  usecols=[0])[0].tolist())
    valid_codes = set([x.lower() for x in valid_codes])
    
    ###### 1. Load GS and Predictions ######
    df_gs = read_gs(gs_path)
    df_run = read_run(pred_path, valid_codes)
    
    ###### 2. Calculate score ######
    P_per_cc, P, R_per_cc, R, F1_per_cc, F1 = calculate_metrics(df_gs, df_run)
    
    return round(P, 3), round(R, 3), round(F1, 3)