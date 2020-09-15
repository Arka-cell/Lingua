import nltk
import spacy
nltk.download('conll2000')

from nltk.corpus import conll2000
from nltk.chunk.util import tree2conlltags,conlltags2tree

from nltk.tag import UnigramTagger, BigramTagger
from nltk.chunk import ChunkParserI
import pandas as pd

data = conll2000.chunked_sents()
train_data = data[:10900]
test_data = data[10900:]

wtc = tree2conlltags(train_data[1])
tree = conlltags2tree(wtc)

def conll_tag_chunks(chunk_sents):
    tagged_sents = [tree2conlltags(tree) for tree in chunk_sents]
    return [[(t, c) for (w, t, c) in sent] for sent in tagged_sents]
def combined_tagger(train_data, taggers, backoff=None):
    for tagger in taggers:
        backoff = tagger(train_data, backoff=backoff)
    return backoff

#Define the chunker class
class NGramTagChunker(ChunkParserI):
    def __init__(self,train_sentences,tagger_classes=[UnigramTagger,BigramTagger]):
        train_sent_tags = conll_tag_chunks(train_sentences)
        self.chunk_tagger = combined_tagger(train_sent_tags,tagger_classes)
    def parse(self,tagged_sentence):
        if not tagged_sentence:
            return None
        pos_tags = [tag for word, tag in tagged_sentence]
        chunk_pos_tags = self.chunk_tagger.tag(pos_tags)
        chunk_tags = [chunk_tag for (pos_tag,chunk_tag) in chunk_pos_tags]
        wpc_tags = [(word,pos_tag,chunk_tag) for ((word,pos_tag),chunk_tag) in zip(tagged_sentence,chunk_tags)]
        return conlltags2tree(wpc_tags)
#train chunker model
ntc = NGramTagChunker(train_data)

class Chunk:

    def __init__(self, text):
        self.text = [sen for sen in text]
        self.nltk_pos_tagged = [nltk.pos_tag(sen.split()) for sen in self.text]
        self.chunk_tree = [ntc.parse(sen) for sen in self.nltk_pos_tagged]
    def tree(self):
        return self.chunk_tree
    def dic(self):
        chunk_dic = {}
        chunk_dics = []
        for tree in self.chunk_tree:
            for chunk, i in zip(tree, range(len(tree))):
                if str(chunk)[1]!= "'":
                    chunk_dic[i] = tuple((str(chunk)[1:3],[data for data in chunk]))
                else:
                    chunk_dic[i] = tuple(("N/A",[data for data in chunk]))
                chunk_dics.append(chunk_dic)
        return chunk_dics
    def dataframe(self):
        i=0 # the iterator will specify the id for the sentence.
        data = self.chunk_tree
        ph = []
        phrases = []
        ph_length = []
        sen_id = []
        for tree in data:
            if tree != None:
                for phrase in tree:
                    words = []
                    if str(phrase)[1] != "'":
                        ph.append(str(phrase)[1:3])
                        for word in phrase:
                            words.append(word[0])
                    phrases.append(words)
                    if len(words)>0:
                        sen_id.append(i)
                i +=1
        phrases = [lst for lst in phrases if lst]
        ph_length = [len(lst) for lst in phrases]
        df = pd.DataFrame({"Sentece ID":sen_id, "Phrases":ph, "Words":phrases, "Phrases Length":ph_length})
        return df
