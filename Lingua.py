import spacy
from spacy import displacy
import pandas as pd
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from chunker import Chunk
from wordfreq import word_frequency

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe(nlp.create_pipe('sentencizer'))

class Linguistics:
    def __init__(self,data):
        self.stringobject = nlp(data)
        self.sentences = [sent.string.strip() for sent in self.stringobject.sents]
        self.sentences = [nlp(sentence) for sentence in self.sentences]
        self.senLength = {}

    def sentences(self):
        return self.sentences

    def punctuation(self):
        self.sentencespunc = []
        for sentence in self.sentences:
            sentencepunc = ""
            string = ""
            for token in sentence:
                if token.is_punct == False:
                    if token != sentence[-1]:
                        sentencepunc = sentencepunc + str(sentence[token.i]) + " "
                    else:
                        sentencepunc = sentencepunc + str(sentence[token.i])
            if len(sentencepunc) > 0 and sentencepunc[-1] == " ":
                sentencepunc = sentencepunc[0:-1]
            self.sentencespunc.append(sentencepunc)
        return [nlp(sentence) for sentence in self.sentencespunc]

    def lengths(self):
        for sentence, n in zip(self.sentences, range(len(self.sentences))):
            self.senLength[n] = [len(sentence)]
        return self.senLength

    def deplengths(self, punc=False):
        N_deps = {} # The whole list of dependecies is notated with a capital n
        n = 0
        # we need dependency either with or without removal of punctuation
        if punc == True:
            self.cleansentences = self.punctuation()
        else:
            self.cleansentences = self.sentences
        for sentence in self.cleansentences:
            n_dep = []
            if type(sentence) != type(True):
                for token in sentence:
                    token_n = [abs(child.i - token.i) for child in token.children]
                    n_dep.append(sum(token_n))
                N_deps[n] = sum(n_dep)
                n+=1
        return N_deps

    def getSentence(self, start_index, end_index=False):
        # If you want to get the whole sentence, you can call sentences method
        if end_index == False:
            return self.sentences[start_index]
        else:
            return self.sentences[start_index:end_index]

    def tree(self, index):
        # In order to display multiple sentences, make an iteration through your
        # list and call Lingua.Linguistics().tree(i)
        return displacy.render(self.sentences[index], style="dep")


    def words(self):
        max_value = int(max(self.lengths().values())[0])
        serie_words = []
        for sentence in self.sentences:
            texts = []
            for token in sentence:
                texts.append(token.text)
            serie_words.append(texts)
        df_words = pd.DataFrame(serie_words, columns=["Word n°"+str(i) for i in range(0,max_value)])
        return df_words


    def pos(self):
        max_value = int(max(self.lengths().values())[0])
        serie_pos = []
        for sentence in self.sentences:
            pos = []
            for token in sentence:
                pos.append(token.pos_)
            serie_pos.append(pos)
        df_pos = pd.DataFrame(serie_pos, columns=["Word n°"+str(i) for i in range(0,max_value)])
        return df_pos

    def dep(self):
        max_value = int(max(self.lengths().values())[0])
        serie_dep = []
        for sentence in self.sentences:
            dep = []
            for token in sentence:
                dep.append(token.dep_)
            serie_dep.append(dep)
        df_dep = pd.DataFrame(serie_dep, columns=["Word n°"+str(i) for i in range(0,max_value)])
        return df_dep

    def postag(self):
        max_value = int(max(self.lengths().values())[0])
        serie_pos = []
        for sentence in self.sentences:
            pos = []
            for token in sentence:
                pos.append(token.tag_)
            serie_pos.append(pos)
        df_postag = pd.DataFrame(serie_pos, columns=["Word n°"+str(i) for i in range(0,max_value)])
        return df_postag

    def freqdist(self, att="tag"):
        if att == "tag":
            df = self.postag()
        elif att == "dep":
            df = self.dep()
        elif att == "pos":
            df = self.pos()
        elif att == "words":
            df = self.words()
        columns = [col for col in df]
        values = []
        for value in [value for value in [df[col] for col in columns]]:
            for i in range(len(df)):
                values.append(value[i])
        df = pd.DataFrame(values, columns=["Values"])
        df.dropna(inplace=True)
        fdist = df["Values"].value_counts()
        return fdist

    def DistPlot(self,att="tag"):
        if att == "tag":
            df = self.freqdist(att="tag")
        elif att == "dep":
            df = self.freqdist(att="dep")
        elif att == "pos":
            df = self.freqdist(att="pos")
        elif att == "words":
            df = self.freqdist(att="words")

        fdist = FreqDist(tag for tag in df)
        fdist.plot(30,cumulative=False)
        return plt.show()

    def string(self):
        return [str(sentence) for sentence in self.sentences]
    def Phrases(self, tree=False):

        chunks = {}
        if tree == False:
            chunks = Chunk(self.string()).dataframe()
            return chunks
        else:
            chunks = Chunk(self.string()).tree()
            return chunks
    def CooConLen(self, punc=False):
        if punc==True:
            sens = self.punctuation()
        else:
            sens = self.sentences
        sen_id = 0
        df = {}
        cc = ["for", "and", "nor", "but", "or", "yet", "so"]
        for sentence in sens:
            df[sen_id] = [0]
            for token in sentence:
                if token.text in cc and token.i > 0:
                    df[sen_id] = [df[sen_id][0]+token.i+(len(sentence) - token.i)]

            sen_id +=1
        return df

    def wordfreq(self):
        values = []
        words = []
        unique_tokens = set([token.text for token in self.stringobject])
        for token in unique_tokens:
            values.append(word_frequency(token,'en'))
            words.append(token)
        df = pd.DataFrame({"Words":words,"Frequency":values})
        return df
