{
    "cells": [
        {
            "cell_type": "code",
            "execution_count": null,
            "id": "05325d1a",
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Logistic Regresion\n",
                        "Akurasi =  0.8871181938911022\n",
                        "Matrix Confussion\n",
                        "[[485   1  66]\n",
                        " [  1 164  19]\n",
                        " [ 69  14 687]]\n",
                        "              precision    recall  f1-score   support\n",
                        "\n",
                        "           1       0.87      0.88      0.88       552\n",
                        "           2       0.92      0.89      0.90       184\n",
                        "           3       0.89      0.89      0.89       770\n",
                        "\n",
                        "    accuracy                           0.89      1506\n",
                        "   macro avg       0.89      0.89      0.89      1506\n",
                        "weighted avg       0.89      0.89      0.89      1506\n",
                        "\n",
                        "K-Neighborhood \n"
                    ]
                },
                {
                    "name": "stderr",
                    "output_type": "stream",
                    "text": [
                        "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\neighbors\\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.\n",
                        "  mode, _ = stats.mode(_y[neigh_ind, k], axis=1)\n"
                    ]
                },
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Akurasi =  0.8399734395750332\n",
                        "Matrix Confussion\n",
                        "[[444  19  89]\n",
                        " [  1 175   8]\n",
                        " [100  24 646]]\n",
                        "              precision    recall  f1-score   support\n",
                        "\n",
                        "           1       0.81      0.80      0.81       552\n",
                        "           2       0.80      0.95      0.87       184\n",
                        "           3       0.87      0.84      0.85       770\n",
                        "\n",
                        "    accuracy                           0.84      1506\n",
                        "   macro avg       0.83      0.86      0.84      1506\n",
                        "weighted avg       0.84      0.84      0.84      1506\n",
                        "\n",
                        "Decision Tree\n",
                        "Akurasi =  0.8193891102257637\n",
                        "Matrix Confussion\n",
                        "[[417   5 130]\n",
                        " [  3 175   6]\n",
                        " [118  10 642]]\n",
                        "              precision    recall  f1-score   support\n",
                        "\n",
                        "           1       0.78      0.76      0.77       552\n",
                        "           2       0.92      0.95      0.94       184\n",
                        "           3       0.83      0.83      0.83       770\n",
                        "\n",
                        "    accuracy                           0.82      1506\n",
                        "   macro avg       0.84      0.85      0.84      1506\n",
                        "weighted avg       0.82      0.82      0.82      1506\n",
                        "\n"
                    ]
                }
            ],
            "source": [
                "# -*- coding: utf-8 -*-\n",
                "\"\"\"\n",
                "Created on Wed Feb  6 15:20:04 2019\n",
                "\n",
                "@author: Agus Nursikuwagus\n",
                "classification tweeter\n",
                "\"\"\"\n",
                "import csv\n",
                "import pandas as pd \n",
                "\n",
                "from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory\n",
                "from sklearn import model_selection\n",
                "from sklearn.metrics import classification_report\n",
                "from sklearn.metrics import confusion_matrix\n",
                "from sklearn.metrics import accuracy_score\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.tree import DecisionTreeClassifier\n",
                "from sklearn.neighbors import KNeighborsClassifier\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer\n",
                "\n",
                "#-------------------------------------------------------------\n",
                "#Reading file from \"Tweet.csv\"\n",
                "def baca_file():\n",
                "    \n",
                "    csvF1 = \"Tweet.csv\"\n",
                "    \n",
                "    #Open file Tweet.csv to manipulate  \n",
                "    with open(csvF1,\"r\") as rCsv:\n",
                "        readCsv = csv.reader(rCsv, delimiter = ';')\n",
                "        read = []\n",
                "        for row in readCsv:\n",
                "            if len(row) != 0:\n",
                "               read = read + [row]\n",
                "                \n",
                "    rCsv.close()\n",
                "    return(read)\n",
                "    \n",
                "#--------------------------------------------------------------\n",
                "#Procedure for displaying the result to the console\n",
                "def tampil_csv(f2):\n",
                "    df3 = pd.DataFrame(f2)\n",
                "    print(df3)\n",
                "    \n",
                "#--------------------------------------------------------------      \n",
                "#Function stemming and return the value of feature and target\n",
                "def stemmingFile(fCsv):\n",
                "    \n",
                "    #---Define a new list for temporary reading---#\n",
                "    rList = []\n",
                "    eList =[]\n",
                "     \n",
                "    #---initialization a stopword by Sastrawi---#\n",
                "    facto  = StopWordRemoverFactory()\n",
                "    stopwords = facto.create_stop_word_remover()\n",
                "        \n",
                "    #---Looping to read line by line csv file---#  \n",
                "    for idx in fCsv:\n",
                "        rList.append(stopwords.remove(idx[0]))\n",
                "        \n",
                "        #---change every word in target to new value---#\n",
                "        if idx[1] == 'Keluhan':\n",
                "            eList.append('1')\n",
                "        elif idx[1]== 'Respon':\n",
                "            eList.append('2')\n",
                "        else:\n",
                "            eList.append('3')\n",
                "        #--- end of IF ---#\n",
                "        \n",
                "    #--- end of looping ---#\n",
                "    return (rList,eList)    #parameter return\n",
                "\n",
                "#-------------------------------------------------------------\n",
                "#procedure to classify every sample in Tweeter.csv\n",
                "def classiLogRegressi(lRead, rRead):\n",
                "\n",
                "    #---setting validation 20% fromm data sample---#\n",
                "    validation_size = 0.20\n",
                "    seed = 7\n",
                "    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(lRead, rRead, test_size=validation_size, random_state=seed)\n",
                "\n",
                "    #---TF-IDF vectorizer, collecting value into vector---#\n",
                "    w = TfidfVectorizer()\n",
                "\n",
                "    print('Logistic Regresion')\n",
                "    logistic = LogisticRegression()\n",
                "    logistic = Pipeline([\n",
                "            ('xPipe',w),\n",
                "            ('knn', logistic)])\n",
                "    \n",
                "    logistic.fit(X_train, Y_train)\n",
                "    predictions = logistic.predict(X_validation)\n",
                "    \n",
                "    print('Akurasi = ', accuracy_score(Y_validation, predictions))\n",
                "    print('Matrix Confussion')\n",
                "    print(confusion_matrix(Y_validation, predictions))\n",
                "    print(classification_report(Y_validation, predictions))\n",
                "    \n",
                "    return(logistic)\n",
                "\n",
                "#------------------------------------------------------------\n",
                "def classKNeighborsClassifier(lRead, rRead):\n",
                "\n",
                "    #---setting validation 20% fromm data sample---#\n",
                "    validation_size = 0.20\n",
                "    seed = 7\n",
                "    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(lRead, rRead, test_size=validation_size, random_state=seed)\n",
                "\n",
                "    #---TF-IDF vectorizer, collecting value into vector---#\n",
                "    w = TfidfVectorizer()\n",
                "#       \n",
                "    #---classification using K-NN---# \n",
                "    print('K-Neighborhood ')\n",
                "    knn = KNeighborsClassifier()\n",
                "    knn = Pipeline([\n",
                "            ('xPipe',w),\n",
                "            ('knn', knn)])\n",
                "    \n",
                "    knn.fit(X_train, Y_train)\n",
                "    predictions = knn.predict(X_validation)\n",
                "    print('Akurasi = ', accuracy_score(Y_validation, predictions))\n",
                "    print('Matrix Confussion')\n",
                "    print(confusion_matrix(Y_validation, predictions))\n",
                "    print(classification_report(Y_validation, predictions))\n",
                "    \n",
                "    return(knn)\n",
                "\n",
                "#-------------------------------------------------------------\n",
                "def classDecisionTree(lRead, rRead):\n",
                "\n",
                "    #---setting validation 20% fromm data sample---#\n",
                "    validation_size = 0.20\n",
                "    seed = 7\n",
                "    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(lRead, rRead, test_size=validation_size, random_state=seed)\n",
                "\n",
                "    #---TF-IDF vectorizer, collecting value into vector---#\n",
                "    w = TfidfVectorizer()\n",
                "#       \n",
                "    #---classification using K-NN---# \n",
                "    print('Decision Tree')\n",
                "    deTree = DecisionTreeClassifier()\n",
                "    deTree = Pipeline([\n",
                "            ('xPipe',w),\n",
                "            ('knn', deTree)])\n",
                "    \n",
                "    deTree.fit(X_train, Y_train)\n",
                "    predictions = deTree.predict(X_validation)\n",
                "    print('Akurasi = ', accuracy_score(Y_validation, predictions))\n",
                "    print('Matrix Confussion')\n",
                "    print(confusion_matrix(Y_validation, predictions))\n",
                "    print(classification_report(Y_validation, predictions))\n",
                "    \n",
                "    return(deTree)\n",
                "\n",
                "\n",
                "#----------------------------------------------------------------    \n",
                "def singleTextLogisticRegression(xText, mknn):   \n",
                "    \n",
                "    x_test =[]\n",
                "    x_test.append(xText)\n",
                "    mpredictions = mknn.predict(x_test)\n",
                "    \n",
                "    return(mpredictions)\n",
                "    \n",
                "#----------------------------------------------------------------\n",
                "        \n",
                "def singleTextKNeighbor(xText, cKboar):   \n",
                "    \n",
                "    x_test =[]\n",
                "    x_test.append(xText)\n",
                "    mpredictions = cKboar.predict(x_test)\n",
                "    \n",
                "    return(mpredictions)\n",
                "\n",
                "#-----------------------------------------------------------------\n",
                "    \n",
                "def singleTextDecisionTree(xText, dTree):   \n",
                "    \n",
                "    x_test =[]\n",
                "    x_test.append(xText)\n",
                "    mpredictions = dTree.predict(x_test)\n",
                "            \n",
                "    return(mpredictions)    \n",
                "    \n",
                "#-----------------------------------------------------------------\n",
                "def singleTextNaiveBayes(xText, mBayes):   \n",
                "    \n",
                "    x_test =[]\n",
                "    x_test.append(xText)\n",
                "    mpredictions = mBayes.predict(x_test)\n",
                "            \n",
                "    return(mpredictions)\n",
                "#-----------------------------------------------------------------\n",
                "    \n",
                "def konversiPrediksi(pre):\n",
                "    tulis = ''\n",
                "    if pre == '1':\n",
                "        tulis = 'Keluhan'\n",
                "    elif pre== '2':\n",
                "        tulis = 'Respon'\n",
                "    else:\n",
                "        tulis = 'Not Keluhan/Respon' \n",
                "    \n",
                "    return(tulis)     \n",
                "            \n",
                "#-----Program utama----------------------------------------------- \n",
                "if __name__ == '__main__': \n",
                "        \n",
                "    dList, fList = stemmingFile(baca_file())\n",
                "    \n",
                "    #---model logistic regression---\n",
                "        \n",
                "    logRes   = classiLogRegressi(dList, fList)\n",
                "    Neighbor = classKNeighborsClassifier(dList, fList)\n",
                "    DesTree  = classDecisionTree(dList, fList)  \n",
                "    \n",
                "          \n",
                "    testing = input('Masukkan text tweet = ')\n",
                "\n",
                "    l = singleTextLogisticRegression(testing, logRes)\n",
                "    \n",
                "    k = singleTextKNeighbor(testing, Neighbor)\n",
                "    \n",
                "    t = singleTextDecisionTree(testing, DesTree)\n",
                "       \n",
                "    \n",
                "    print('Prediksi dengan Logistic Regression = ',konversiPrediksi(l))\n",
                "    print('Prediksi dengan K-Nearest Neighirhood =',konversiPrediksi(k))\n",
                "    print('Prediksi dengan Decision Tree = ', konversiPrediksi(t))\n",
                "        \n",
                "\n",
                "  \n",
                "    \n",
                "#----End of Program-------------------------------------------------\n",
                "\n",
                " \n",
                "    \n",
                "    \n",
                "       \n",
                "   \n",
                "   \n",
                "    \n",
                "    \n",
                "    \n",
                "\n",
                "\n",
                "\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "id": "43a9627f",
            "metadata": {},
            "outputs": [],
            "source": []
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}