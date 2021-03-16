### Module containing auxiliary functions and classes for NLP using BERT


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
    It loads the start-end pair of each splitted sentence from a set of files (start + \t + end line format expected) into a 
    returned dictionary, where keys are file names and values a list of tuples conatining the start-end pairs of the 
    splitted sentences.
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
    
    Check the annotations contained in the files are in BRAT format with codes assigned.
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


def convert_token_to_id_attention(token_list, tokenizer, seq_len):
    """
    Given a list of SP sub-tokens ids representing a sentence, and a tokenizer, it returns their correponding lists of 
    indices and attention masks. Padding is added as appropriate.
    """
    
    # Convert SP ids to correponding model ids
    for i in range(len(token_list)):
        token_list[i] = token_list[i] + tokenizer.fairseq_offset if token_list[i] else tokenizer.unk_token_id
    
    # Add [CLS] and [SEP] tokens
    token_ids = [tokenizer.cls_token_id] + token_list + [tokenizer.sep_token_id]
    
    # Generate attention mask
    first_len = len(token_ids)
    attention_mask = [1] * first_len
    
    # Padding
    pad_len = seq_len - first_len
    token_ids += [tokenizer.pad_token_id] * pad_len
    attention_mask += [0] * pad_len

    return token_ids, attention_mask


def start_end_tokenize(text, sp_model, sp_pb2, start_byte=0):
    """
    Our aim is to produce both a list of SP sub-tokens ids and a list of tuples containing the bytes start-end offset
    of each sub-token.
    
    start_byte: the byte start position of the first character in the text.
    """
    
    sp_pb2.ParseFromString(sp_model.encode_as_serialized_proto(text))
    
    tokens_id, start_end_arr = [], []
    for p in sp_pb2.pieces:
        tokens_id.append(p.id)
        start_end_arr.append((p.begin + start_byte, p.end + start_byte))
    
    return tokens_id, start_end_arr



def ss_start_end_tokenize(ss_start_end, max_seq_len, text, text_char_bytes, sp_model, sp_pb2):
    """
    ss_start_end: list of tuples, where each tuple contains the start-end character positions pair of 
                  the splitted sentences from the input document text.
    text: document text
    
    return: two lists of lists, the first for the SentencePiece (SP) sub-tokens ids from the re-split sentences and the second for the 
            sub-tokens bytes start-end offset pairs.
    """
    out_sub_token_id, out_start_end = [], []
    for ss_start, ss_end in ss_start_end:
        sub_token_id, start_end = start_end_tokenize(text=text[ss_start:ss_end], sp_model=sp_model, sp_pb2=sp_pb2,
                                                     start_byte=text_char_bytes[ss_start][0])
        assert len(sub_token_id) == len(start_end)
        # Re-split large sub-tokens splitted sentences
        for i in range(0, len(sub_token_id), max_seq_len):
            out_sub_token_id.append(sub_token_id[i:i+max_seq_len])
            out_start_end.append(start_end[i:i+max_seq_len])
    
    return out_sub_token_id, out_start_end


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
    
        

## Brute force approach to annotate each fragment

from tqdm import tqdm
from copy import deepcopy

def ss_brute_force_create_frag_input_data(df_text, text_col, df_ann, doc_list, ss_dict, tokenizer, sp_pb2, lab_encoder, seq_len, 
                                          encoding='utf-8', err_arg='strict'):
    """
    ss_dict: dict where keys are file names and values a list of tuples containing the char start-end pairs of the 
    splitted sentences in each file.
    
    Temporal complexity: O(n_doc x n_frag x n_ann), where n_frag and n_ann vary for each doc.
    """
    indices, attention_mask, labels, n_fragments, start_end_offsets = [], [], [], [], []
    for doc in tqdm(doc_list):
        # Extract doc annotation
        doc_ann = df_ann[df_ann["doc_id"] == doc]
        # Extract doc text
        doc_text = df_text[df_text["doc_id"] == doc][text_col].values[0]
        # Extract bytes start-end offset of each character in the text
        ch_start_end_bytes = convert_char_to_bytes_offset(doc_text, encoding, err_arg)
        
        ## Generate fragments
        # Perform SS of doc text
        doc_ss = ss_dict[doc] # SS start-end pairs of the doc
        doc_ss_token_id, doc_ss_start_end = ss_start_end_tokenize(ss_start_end=doc_ss, 
                                    max_seq_len=seq_len-2, text=doc_text, text_char_bytes=ch_start_end_bytes,
                                    sp_model=tokenizer.sp_model, sp_pb2=sp_pb2)
        assert len(doc_ss_token_id) == len(doc_ss_start_end)
        frag_token = deepcopy(doc_ss_token_id)
        frag_start_end = deepcopy(doc_ss_start_end)
                
        assert len(frag_token) == len(frag_start_end)
        # Store the start-end char positions of all the fragments
        start_end_offsets.extend(frag_start_end)
        # Store the number of fragments of each doc text
        n_fragments.append(len(frag_token))
        
        ## Assign to each fragment the labels (codes) from the NER-annotations exclusively occurring inside 
        # the fragment
        for f_token, f_start_end in zip(frag_token, frag_start_end):
            # fragment length is assumed to be <= SEQ_LEN-2
            assert len(f_token) == len(f_start_end) <= seq_len-2
            # Indices & Segments
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


# As all abstracts have one single fragment, the next method is compatible with NER method (previous function) to generate data

def create_frag_input_data(df_text, text_col, df_label, doc_list, tokenizer, sp_pb2, lab_encoder, seq_len):
    indices, attention_mask, labels, fragments = [], [], [], []
    for doc in tqdm(doc_list):
        # Extract labels
        doc_labels = list(df_label[df_label["doc_id"] == doc]["code"])
        # Tokenize doc text into a list of tokens
        doc_token, _ = start_end_tokenize(text=df_text[df_text["doc_id"] == doc][text_col].values[0], 
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



# MAP score evaluation

def max_fragment_prediction(y_frag_pred, n_fragments, label_encoder_classes, doc_list):
    """
    Convert fragment-level to doc-level predictions, usin max criterion.
    """
    y_pred = []
    i_frag = 0
    for n_frag in n_fragments:
        y_pred.append(y_frag_pred[i_frag:i_frag+n_frag].max(axis=0))
        i_frag += n_frag
    return prob_codiesp_prediction_format(np.array(y_pred), label_encoder_classes, doc_list)


def mean_fragment_prediction(y_frag_pred, n_fragments, label_encoder_classes, doc_list):
    """
    Convert fragment-level to doc-level predictions, usin mean criterion.
    """
    y_pred = []
    i_frag = 0
    for n_frag in n_fragments:
        y_pred.append(y_frag_pred[i_frag:i_frag+n_frag].mean(axis=0))
        i_frag += n_frag
    return prob_codiesp_prediction_format(np.array(y_pred), label_encoder_classes, doc_list)


def max_mean_fragment_prediction(y_frag_pred, n_fragments, label_encoder_classes, doc_list):
    """
    Convert fragment-level to doc-level predictions, usin max*mean criterion.
    """
    y_pred = []
    i_frag = 0
    for n_frag in n_fragments:
        y_pred.append(np.multiply(y_frag_pred[i_frag:i_frag+n_frag].max(axis=0),y_frag_pred[i_frag:i_frag+n_frag].mean(axis=0)))
        i_frag += n_frag
    return prob_codiesp_prediction_format(np.array(y_pred), label_encoder_classes, doc_list)


def clinbert_fragment_prediction(y_frag_pred, n_fragments, label_encoder_classes, doc_list, c=2):
    """
    Convert fragment-level to doc-level predictions, usin ClinicalBERT criterion.
    """
    y_pred = []
    i_frag = 0
    for n_frag in n_fragments:
        p_max = y_frag_pred[i_frag:i_frag+n_frag].max(axis=0)
        p_mean = y_frag_pred[i_frag:i_frag+n_frag].mean(axis=0)*n_frag/c
        y_pred.append((p_max + p_mean)/(1+n_frag/c))
        i_frag += n_frag
    return prob_codiesp_prediction_format(np.array(y_pred), label_encoder_classes, doc_list)


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



from tensorflow.keras.callbacks import Callback

class EarlyMAP_Frag(Callback):
    """
    Custom callback that performs early-stopping strategy monitoring MAP-prob metric on validation fragment dataset.
    Both train and validation MAP-prob values are reported in each epoch.
    """
    
    def __init__(self, x_train, x_val, frag_train, frag_val, label_encoder_cls, valid_codes, train_doc_list, val_doc_list, 
                 train_gs_file=None, val_gs_file=None, patience=10):
        self.X_train = x_train
        self.X_val = x_val
        self.frag_train = frag_train
        self.frag_val = frag_val
        self.label_encoder_cls = label_encoder_cls
        self.valid_codes = valid_codes
        self.train_doc_list = train_doc_list
        self.val_doc_list = val_doc_list
        self.train_gs_file = train_gs_file
        self.val_gs_file = val_gs_file
        self.patience = patience
    
    
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None


    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## MAP-prob
        ### Train data
        y_pred_train = self.model.predict(self.X_train)
        # Save predictions file in CodiEsp format
        df_pred_train = max_fragment_prediction(y_frag_pred=y_pred_train, n_fragments=self.frag_train, 
                                                label_encoder_classes=self.label_encoder_cls, 
                                                doc_list=self.train_doc_list)
        map_train = compute_map(valid_codes=self.valid_codes, pred=df_pred_train, gs_out_path=self.train_gs_file)
        logs['map'] = map_train
        
        ### Val data
        y_pred_val = self.model.predict(self.X_val)
        # Save predictions file in CodiEsp format
        df_pred_val = max_fragment_prediction(y_frag_pred=y_pred_val, n_fragments=self.frag_val, 
                                              label_encoder_classes=self.label_encoder_cls, 
                                              doc_list=self.val_doc_list)
        map_val = compute_map(valid_codes=self.valid_codes, pred=df_pred_val, gs_out_path=self.val_gs_file)
        logs['val_map'] = map_val            
        
        # Early-stopping
        if (map_val > self.best):
            self.best = map_val
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)