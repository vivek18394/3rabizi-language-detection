#arabizi-english-bitext.txt -> preprocess.txt -> arabizi_vocab -> final_vocab
import re
import os
import nltk
import string

def preprocess(source_file):
    print(source_file)
    input = open(source_file)
    preprocess_write = open("preprocess.txt","w")
    for line in input:
        preprocess_write.write(re.sub(r'(.)\1+', r'\1\1', line))
    input.close()
    preprocess_write.close()

    

#Tokenize the preprocessed text file
def tokenize(file_name):
    print(file_name)
    file_content = open(file_name).read()
    tokens = nltk.word_tokenize(file_content)
    filter_token = [t for t in tokens if t not in string.punctuation]#Remove punctutation
    filtered_words = [w.lower for w in filter_token]# Lower case
    filtered_vocab = sorted(set(filter_token))
    with open("arabizi_vocab.txt", "w") as f:
        for item in filtered_vocab:
            f.write(item + "\n")
    print("The number of words in vocabulary is ",len(filtered_vocab))
    os.remove("preprocess.txt")

def remove_num_vocab():
    with open("arabizi_vocab.txt","r") as input:
        with open("final_vocab.txt","w") as output: 
            for line in input:
                line = line.strip() # clear white space
                try:
                    int(line) #is this a number ?
                except ValueError:
                    output.write(line + "\n")
    Strip_Punctuation()
    os.remove("arabizi_vocab.txt")

def Strip_Punctuation():
    translator = str.maketrans('', '', string.punctuation)
    with open("final_vocab.txt") as f_in:
        with open("vocabulary.txt","w") as f_out:
            for line in f_in:
                #line.translate(translator)
                #content = f_in.readlines()
                #content.translate(None, string.punctuation)
                f_out.write(line.translate(translator))
    os.remove("final_vocab.txt")

#Main function
preprocess("arabizi-english-bitext.arz")
tokenize("preprocess.txt")
remove_num_vocab()
print("Done")