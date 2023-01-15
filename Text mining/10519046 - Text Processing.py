#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
import pandas as pd 

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# In[9]:


#Reading file from \"Tweet.csv\"
def baca_file():
    
    csvF1 = "Tweet.csv"
    
    #Open file Tweet.csv to manipulate  
    with open(csvF1,"r") as rCsv:
        readCsv = csv.reader(rCsv, delimiter = ';')
        read = []
        for row in readCsv:
            if len(row) != 0:
               read = read + [row]
                
    rCsv.close()
    return(read)


# In[10]:


def tampil_csv(f2):
        df3 = pd.DataFrame(f2)
        print(df3)


# In[12]:


#Function stemming and return the value of feature and target\n",
def stemmingFile(fCsv):
        #---Define a new list for temporary reading---#
        rList = []
        eList =[]
         
        #---initialization a stopword by Sastrawi---#
        facto  = StopWordRemoverFactory()
        stopwords = facto.create_stop_word_remover()
            
        #---Looping to read line by line csv file---#  
        for idx in fCsv:
            rList.append(stopwords.remove(idx[0]))
            
            #---change every word in target to new value---#
            if idx[1] == 'Keluhan':
                eList.append('1')
            elif idx[1]== 'Respon':
                eList.append('2')
            else:
                eList.append('3')
            #--- end of IF ---#
            
        #--- end of looping ---#
        return (rList,eList)    #parameter return


# In[13]:


def classiLogRegressi(lRead, rRead):

    #---setting validation 20% fromm data sample---#
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(lRead, rRead, test_size=validation_size, random_state=seed)

    #---TF-IDF vectorizer, collecting value into vector---#
    w = TfidfVectorizer()

    print('Logistic Regresion')
    logistic = LogisticRegression()
    logistic = Pipeline([
            ('xPipe',w),
            ('knn', logistic)])
    
    logistic.fit(X_train, Y_train)
    predictions = logistic.predict(X_validation)
    
    print('Akurasi = ', accuracy_score(Y_validation, predictions))
    print('Matrix Confussion')
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    
    return(logistic)


# In[14]:


def classKNeighborsClassifier(lRead, rRead):
    #---setting validation 20% fromm data sample---#
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(lRead, rRead, test_size=validation_size, random_state=seed)

    #---TF-IDF vectorizer, collecting value into vector---#
    w = TfidfVectorizer()
#       
    #---classification using K-NN---# 
    print('K-Neighborhood ')
    knn = KNeighborsClassifier()
    knn = Pipeline([
            ('xPipe',w),
            ('knn', knn)])
    
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print('Akurasi = ', accuracy_score(Y_validation, predictions))
    print('Matrix Confussion')
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    
    return(knn)


# In[15]:


def classDecisionTree(lRead, rRead):
    #---setting validation 20% fromm data sample---#
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(lRead, rRead, test_size=validation_size, random_state=seed)

    #---TF-IDF vectorizer, collecting value into vector---#
    w = TfidfVectorizer()
    #---classification using K-NN---# 
    print('Decision Tree')
    deTree = DecisionTreeClassifier()
    deTree = Pipeline([
            ('xPipe',w),
            ('knn', deTree)])
    
    deTree.fit(X_train, Y_train)
    predictions = deTree.predict(X_validation)
    print('Akurasi = ', accuracy_score(Y_validation, predictions))
    print('Matrix Confussion')
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))
    
    return(deTree)


# In[16]:


def singleTextLogisticRegression(xText, mknn):
        x_test =[]
        x_test.append(xText)
        mpredictions = mknn.predict(x_test)
        
        return(mpredictions)


# In[17]:


def singleTextKNeighbor(xText, cKboar):   
    x_test =[]
    x_test.append(xText)
    mpredictions = cKboar.predict(x_test)
    
    return(mpredictions)


# In[18]:


def singleTextDecisionTree(xText, dTree):   
    x_test =[]
    x_test.append(xText)
    mpredictions = dTree.predict(x_test)
            
    return(mpredictions)


# In[19]:


def singleTextNaiveBayes(xText, mBayes):   
    x_test =[]
    x_test.append(xText)
    mpredictions = mBayes.predict(x_test)
            
    return(mpredictions)


# In[20]:


def konversiPrediksi(pre):
    tulis = ''
    if pre == '1':
        tulis = 'Keluhan'
    elif pre== '2':
        tulis = 'Respon'
    else:
        tulis = 'Not Keluhan/Respon' 
    
    return(tulis)


# In[21]:


#Program utama
if __name__ == '__main__': 
        
    dList, fList = stemmingFile(baca_file())
    
    #---model logistic regression---
        
    logRes   = classiLogRegressi(dList, fList)
    Neighbor = classKNeighborsClassifier(dList, fList)
    DesTree  = classDecisionTree(dList, fList)  
    
            
    testing = input('Masukkan text tweet = ')

    l = singleTextLogisticRegression(testing, logRes)
    
    k = singleTextKNeighbor(testing, Neighbor)
    
    t = singleTextDecisionTree(testing, DesTree)
        
    
    print('Prediksi dengan Logistic Regression = ',konversiPrediksi(l))
    print('Prediksi dengan K-Nearest Neighirhood =',konversiPrediksi(k))
    print('Prediksi dengan Decision Tree = ', konversiPrediksi(t))
#End of Program


# In[ ]:




