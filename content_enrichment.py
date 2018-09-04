import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize

def read_file(file):
    with open(file) as f:
        return f.read()
#s="sun is a stars. It is at the centre of solar system. moon is a sateliites. And it is the satelite of earth. xyz is good. she is an engineer"

def content_enrichment(s):
    sent=nltk.sent_tokenize(s)
    for i in range(0,len(sent)-1):
        pos=nltk.pos_tag(nltk.word_tokenize(sent[i]))
        pos=[list(item) for item in list(pos)]
        for j in range(0,5):
            if pos[j][1]=='NN':
                s_old=pos[j][0]
                s_new=nltk.pos_tag(nltk.word_tokenize(sent[i+1]))
                s_new=list(s_new)
                s_new=[list(item) for item in s_new]
                for k in range(0,5):
                    if s_new[k][1]=='PRP':
                        s_new[k][0]=s_old
                        #sent[i+1]=s_new
                        print(s_new)
                       #"".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


content_enrichment(read_file('/home/renimol/Desktop/context/FRIDAY/ce'))
                          
    














#pos=nltk.pos_tag(tokens)
#print pos
#df=pd.DataFrame(list(pos))
#print(df)
#print(df[1][5])
