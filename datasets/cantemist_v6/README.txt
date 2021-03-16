Cantemist shared task dataset (divided in train, dev1, dev2 and test). In addition, we include here the Cantemist background set.

It contains the train, development and test sets of the three subtasks: cantemist-ner, cantemist-norm and cantemist-coding with Gold Standard annotations.

In addition, it contains the documents background sets, without annotations.

For subtasks cantemist-norm and cantemist-ner, annotations are distributed in Brat format. See Brat webpage for more information https://brat.nlplab.org/standoff.html

For subtask cantemist-coding, codes are grouped in a TSV file with the following columns (this follows the format used in CodiEsp shared task  https://temu.bsc.es/codiesp/): 
filename	code

In the three subtasks, the goal is to predict the annotations of the test files (either the ANN files or the TSV with the codes) given only the plain text files. 

For further information, please visit https://temu.bsc.es/cantemist/ or email us at encargo-pln-life@bsc.es

