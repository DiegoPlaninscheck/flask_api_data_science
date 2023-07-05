import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import pandas as pd

# Faz o download do tokenizador (Usado para dividir o texto em suas palavras)
nltk.download('punkt')
# Faz o download dos StopWords (Palavras desnecessárias para a análise. Exemplo: a, de, um, ...)
nltk.download('stopwords')

def preprocess_text(sentence):
    # Remove as pontuações de um texto
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    # Transforma as letra para minúsculo
    sentence = sentence.lower()
    # Cria um vetor de palavras
    words = word_tokenize(sentence)
    # Usa os StopWords de todos os português
    stop_words = set(stopwords.words('portuguese'))
    # Remove do vetor as palavras do StopWords
    words = [word for word in words if word not in stop_words]
    # Instancia objeto Stemmer
    stemmer = PorterStemmer()
    # Transforma a palavra para sua forma primitiva (Exemplo: andando --> andar, comeu --> comer)
    words = [stemmer.stem(word) for word in words]
    # Junta as palavras novamente para uma "frase"
    sentece = ' '.join(words)
    return sentece

train_data = [
    ('Eu amo este produto!', 'positivo'), # TF-IDF = 1, 1, 0.5, 0.5
    ('Este produto é horrível!', 'negativo'),
    ('O filme foi incrível', 'positivo'),
    ('Não gostei do serviço', 'negativo')
]

# TF = Quantidade de ocorrências de uma palavra no texto dividido pela quantidade de palavras do texto (Exemplo da palavra porduto: 1 / 4)
# IDF = Quantidade de textos dividido pela quantidade de documentos que contém a palvra em análise (Exemplo da palavra produto: 4 / 2)
# TF * IDF = Para cada texto haverá um vetor onde a palavra será identificada pelo peso dado dessa multiplicação
vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
# Vetoriza as frases para um formato comprensível para treinamento
train_features = vectorizer.fit_transform([x[0] for x in train_data])
# Cria um vetor com os sentimentos de cada texto
train_labels = [x[1] for x in train_data]

#
classifier = svm.SVC(kernel='linear')
# Treina o SVC para aumentar a margem entre os vetores
classifier.fit(train_features, train_labels)

palavras = vectorizer.get_feature_names_out()
# print(pd.DataFrame(palavras))
# print(train_features.toarray())

# Classifica o sentimento de um texto
def predict_sentiment(sentence):
    # Realiza o pré-processamento do texto
    sentece = preprocess_text(sentence)
    # Vetoriza o texto pré-processado
    features = vectorizer.transform([sentence])
    # Classifica o sentimento de acordo com o texto vetorizado
    sentiment = classifier.predict(features)[0]
    return sentiment

# Segundo código:
# text = "Este carro é horrível!"
# sentiment = predict_sentiment(text)
# print(f"Sentimento: {sentiment}")
