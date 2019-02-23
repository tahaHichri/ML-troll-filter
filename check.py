import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

# load features matrix/labels
train_matrix = np.load('enron_features_matrix.npy')
labels = np.load('enron_labels.npy')
dict = np.load('dict_enron.npy')



def extract_features(msg_str):
    docID = 0
    features_matrix = np.zeros((1,3000))
    all_words = []
    words = msg_str.split()
    all_words += words
    for word in all_words:
        wordID = 0
        for i,d in enumerate(dict):
            if d[0] == word:
                wordID = i
                # the number of occurrences of a substring in the given string
                features_matrix[docID,wordID] = all_words.count(word)
    return features_matrix



X_train, X_test, y_train, y_test = train_test_split(train_matrix, labels, test_size=0.40)


naiveBayesM = MultinomialNB()

naiveBayesM.fit(X_train,y_train)



# make a guess
prediction_res= naiveBayesM.predict(extract_features('hello I like this website, the tutorial was really helpful'))

print(prediction_res)



#print (confusion_matrix(y_test, prediction_res))