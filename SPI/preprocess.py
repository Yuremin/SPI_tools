import os,re
from gensim.models import Word2Vec
import numpy as np






def generate_embedding(path):
    """
    
    """
    word2vec4text_model = Word2Vec.load("saved_models/w2v/word2vec4text.model")
    word2vec4code_model = Word2Vec.load("saved_models/w2v/word2vec4code.model")
    vocabulary4text = word2vec4text_model.wv.key_to_index
    vocabulary4code = word2vec4code_model.wv.key_to_index
    
    file_list = []
    for root, dirs, fs in os.walk(path):
        for f in fs:
            if f != 'commit':continue
            file_list.append(os.path.join(root,f))
    
    pattern_message = r"commit [a-f0-9]{40}\n(?:.|\n)*?\n\n([\s\S]*?)(?=\ndiff)"
    for fname in file_list:
        with open(fname, "r", errors='replace') as f:
            lines = f.readlines()
            both_code = []
            addtiva_code = []
            subtractive_code = []
            line_sum = 0
            matched_message = re.search(pattern_message, "".join(lines))
            if matched_message: commit_message = matched_message.group(1).strip().split()
            else: print("nononon") ;break
            
            for line in lines:
                if len(line) <2: continue
                if (line.startswith('+') or line.startswith('-')) and (line[1]!='+' and line[1]!='-'):
                    if line_sum == 10:break
                    line_sum += 1
                    both_code.append(line[1:].split())
                    if line[0] == '+': addtiva_code.append(line[1:].split())
                    elif line[0] == '-': subtractive_code.append(line[1:].split())
                    else:break
            both_code = [j for i in both_code for j in i]
        
        if len(both_code) == 0: both_code = [""]
        filtered_commit_message = [word for word in commit_message if word in vocabulary4text]
        filtered_code = [word for word in both_code if word in vocabulary4code]

        if len(filtered_code) == 0:
            code_vector = [[0]*300]
        else:
            code_vector = word2vec4code_model.wv[filtered_code]
        message_vector = word2vec4text_model.wv[filtered_commit_message]
        label = 1 if "/yes/" in fname else 0
        vector_file = fname.replace('commit', 'vector')
        with open(vector_file, "wb") as f:
            np.savez(vector_file, code_vector=code_vector, message_vector=message_vector, label=label, dtype=object)
    return 


if __name__ == '__main__':
    generate_embedding("data/detection/spidb")