import pandas as pd
import scipy
from sklearn.feature_extraction.text import CountVectorizer

# import nltk
# nltk.download()
vec = CountVectorizer()
def read_file_from_folder(mypath):
    from os import listdir
    from os.path import isfile, join
    return (join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)))

def read_line_from_file(file):
    with open(file) as fp:
        line_number = 0
        for line in fp:
            line_number += 1
            yield file, line_number, line

def read_words_from_line(aLine):
    from nltk.tokenize import word_tokenize
    return word_tokenize(aLine)

def get_tfids(docs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = vec.fit_transform(docs)
    vectorizer = TfidfVectorizer()
    return TfidfVectorizer().fit_transform(docs).todense()

def compute_cutoff(dff):
    return 0.

def apply_tranformation(dff, cutoff):
    return dff.apply(lambda x:([0 if x[i] < cutoff else x[i] for i in range(len(x))]))

def find_docs_curresponding_to_influencial_words(dff):
    dict={}
    for columns in dff.columns:
        for i in range(0,3):
            if dff.iloc[i][columns] > 0:
                dict.setdefault(columns,[])
                dict[columns].append(i+1)         
    return dict

def generate_map(dict):
    print("Influencial words\tDocument No\n")            
    for i in dict:
        print("{}\t\t\t{}".format(i,dict[i]) ) 

def prepare_data(location):
    from nltk.stem import SnowballStemmer
    from nltk.corpus import stopwords

    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))

    doc_word_dic = {}
    bag_of_words = {}
    for aFile in read_file_from_folder(location):
        bag_of_words_per_doc = {}
        doc_words = set(())
        for (fileName, lineNumber, aLine) in (read_line_from_file(aFile)):
            good_words = [word.lower() for word in read_words_from_line(aLine) if not word in stop_words]
            for stemWord in (stemmer.stem(word) for word in good_words):
                bag_of_words_per_doc[stemWord] = bag_of_words_per_doc.get(stemWord, 0) + 1
                bag_of_words[stemWord] = bag_of_words.get(stemWord, 0) + 1
                doc_words.add(stemWord)
            doc_word_dic[fileName] = doc_word_dic.get(fileName, set()).union(doc_words)

    return [' '.join(words) for words in doc_word_dic.values()]

matrix = get_tfids(prepare_data('/home/renimol/Desktop/context aware/FRIDAY/docss'));
dff = pd.DataFrame(matrix, columns=vec.get_feature_names())
print(apply_tranformation(dff, compute_cutoff(dff)))
#print(plot_histogram(dff))
print(generate_map(find_docs_curresponding_to_influencial_words(dff)))





  


