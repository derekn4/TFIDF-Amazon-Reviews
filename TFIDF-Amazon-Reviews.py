'''
LIBRARIES TO INSTALL BEFORE RUNNING (pip install):

1. pandas
2. numpy
3. nltk
4. bs4
5. sklearn
6. contractions
'''
import subprocess
subprocess.call(['pip', 'install', "-q", "pandas"])
subprocess.call(['pip', 'install', "-q", "numpy"])
subprocess.call(['pip', 'install', "-q", "nltk"])
subprocess.call(['pip', 'install', "-q", "bs4"])
subprocess.call(['pip', 'install', "-q", "sklearn"])
subprocess.call(['pip', 'install', "-q", "contractions"])

# In[2]:
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet', quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download('omw-1.4', quiet=True)
from bs4 import BeautifulSoup
import re
import csv

import warnings
warnings.simplefilter("ignore")

# ## Read Data
# In[5]:
'''
Data is pre-loaded into the same folder and is read with pandas.read_csv function.
Parameters identify the headers at row=0, removes bad lines in the file, removes quotations, 
and takes only the review_body and star_rating columms.
'''
try:
    data = pd.read_csv('data.tsv', sep='\t', header=0,
                       error_bad_lines=False, quoting=csv.QUOTE_NONE, usecols=['review_body', 'star_rating'])
except:
    data = pd.read_csv('https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Jewelry_v1_00.tsv.gz', sep='\t', header=0, error_bad_lines = False,
                   quoting=csv.QUOTE_NONE, usecols = ['review_body','star_rating'])


# ## Keep Reviews and Ratings
# In[7]:
'''
Takes the panda data and creates a dataframe that filters out any review_body
cell that is null/empty.
'''
df = pd.DataFrame(data)
df = df[df['review_body'].notnull()]


#  ## We select 20000 reviews randomly from each rating class.
# In[68]:
'''
Selects 20,000 reviews randomly from each star_rating class 1-5
'''
five_star = df.query("star_rating==5").sample(n=20000)
four_star = df.query("star_rating==4").sample(n=20000)
three_star= df.query("star_rating==3").sample(n=20000)
two_star  = df.query("star_rating==2").sample(n=20000) 
one_star  = df.query("star_rating==1").sample(n=20000)


# In[69]:
'''
Merges the random sampling of 20,000 reviews from each rating class into 1 dataframe
'''
balance = pd.concat([five_star, four_star, three_star, two_star, one_star]).reset_index(drop=True)


# # Data Cleaning
# # Pre-processing


# In[72]:
'''
Data cleaning: changes all review_body cells to lowercase and strips '\n'.
Expands Contractions with the "contractions" library (ex: don't -> do not).
Removes html/url tags, accented characters, and any non-alphabetical characters.
Removes any punctuation and extra white spaces in the data.
'''
review_count_clean = len(balance)
review_length_clean_before = balance['review_body'].str.len()
avg_before_clean = review_length_clean_before.sum() // review_count_clean

#Change to lower case + extra spaces on sides
balance["review_body"] = balance["review_body"].str.replace('\n',' ').str.lower()    

#remove contractions (and slang?)
import contractions
balance["review_body"] = balance["review_body"].apply(lambda x: [contractions.fix(word, slang=True) for word in x.split()])
balance["review_body"] = [' '.join(map(str, l)) for l in balance["review_body"]]

def remove_html_tags_func(text):
    return BeautifulSoup(text, 'html.parser').get_text()
balance["review_body"] = balance["review_body"].apply(remove_html_tags_func)

def remove_url_func(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)
balance["review_body"] = balance["review_body"].apply(remove_url_func)

def remove_irr_char_func(text):
    return re.sub(r'[^a-zA-Z]', ' ', text)
balance["review_body"] = balance["review_body"].apply(remove_irr_char_func)


#removing accents
import unicodedata
def remove_accented_chars_func(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
balance["review_body"] = balance["review_body"].apply(remove_accented_chars_func)

#remove non-alphabetical chars
balance["review_body"] = balance["review_body"].str.replace(r'[^a-z]', ' ', regex=True)

#remove html/urls
balance["review_body"] = balance["review_body"].str.replace(r'https?://\S+|www\.\S+','', regex=True)

#remove punctuation
balance["review_body"] = balance["review_body"].str.replace(r'[^\w\s]', '', regex=True)

#remove extra spaces
def remove_extra_whitespaces_func(text):
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()
# balance["review_body"] = balance["review_body"].str.replace(r'\s+', ' ', regex=True)
balance["review_body"] = balance["review_body"].apply(remove_extra_whitespaces_func)

review_length_clean_after = balance['review_body'].str.len()
avg_after_clean = review_length_clean_after.sum() // review_count_clean
print(f"{int(avg_before_clean)},{int(avg_after_clean)}")

# In[74]:
svm_balance = balance.copy()

# ## remove the stop words
# In[80]:
"""
Chose to not remove stop words from Perceptron, SVM, Logistic Regression, and Multinominal Naive Bayes models because I found
that the precision, recall, and f1 scores were higher when I included stop words.

Also, Piazza post @116 (https://piazza.com/class/l7102doc7aa3ob/post/116) said that we were allowed to use
different preprocessing steps for our algorithms.
"""


# In[76]:
'''
Pulls stop words from nltk library and removes all stop words that exist in the review_body column.
'''
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

review_count = len(svm_balance)
review_length = svm_balance['review_body'].str.len()
avg_before = review_length.sum() // review_count

pat = r'\b(?:{})\b'.format('|'.join(stops))
svm_balance['review_body'] = svm_balance['review_body'].str.replace(pat, '')
svm_balance['review_body'] = svm_balance['review_body'].str.replace(r'\s+', ' ')


# ## perform lemmatization  

# In[81]:
'''
Uses WordNetLemmatizer from nltk library and executes lemmatization on all words in the review_body column.
Perceptron, SVM, Logistic Regression, and Multinominal Naive Bayes models all improved by performing lemmization
and including stop words.
EX: runs, running, ran -> run
'''
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

balance["review_body"] = balance["review_body"].apply(lambda x: [wnl.lemmatize(word) for word in x.split()])
balance["review_body"] = [' '.join(map(str, l)) for l in balance["review_body"]]

# In[78]:
#Peform removing stop words and lemmatization and report average lengths before and after.
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

svm_balance["review_body"] = svm_balance["review_body"].apply(lambda x: [wnl.lemmatize(word) for word in x.split()])
svm_balance["review_body"] = [' '.join(map(str, l)) for l in svm_balance["review_body"]]

review_len_after_lem = svm_balance['review_body'].str.len()
avg_after_lem = review_len_after_lem.sum() // review_count
print(f"{int(avg_before)},{int(avg_after_lem)}")


# # TF-IDF Feature Extraction
# In[79]:
from sklearn.feature_extraction.text import TfidfVectorizer

'''
TFIDFVectorizer transforms data into a sparse matrix in order to train ML models.
By Vectorizing and scoring the text in reviews, the ML models can learn by using the tfidf scores.

The TFIDFVectorizer parameters min_df and ngram_range allow for the function 
to remove any words that appear in 10 or less documents (min_df) and create a more diverse dataset with ngram_range.
The ngram_range can take a subset of length 1 to 4 of words that may appear together in success and score the 
tfidf of the phrase rather than just a single word, but without min_df, the dataset can grow exponentially with
ngram_range.
'''
vectorizer = TfidfVectorizer(min_df = 10, ngram_range=(1, 4))

review_list = balance["review_body"].tolist()

vectors = vectorizer.fit_transform(review_list)


# # Split Data Training/Test
# In[82]:
'''
Split the data 80/20 for training and testing sets.

The tfidf vectors created in the last step are used as the training data and the star_ratings are used as the labels.
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(vectors, balance["star_rating"], test_size=0.20, random_state=0)


# In[83]:
#Report Precision, Recall, f1_score per class, AND averages on testing split of dataset
#These 18 values should be printed in separate lines by the .py file.

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# # Perceptron

# In[95]:
'''
Perceptron model is from the sklearn library and uses the parameter "n_iter_no_change" 
which means that the model will keep training epochs over the training data until there are
10 iterations without a change in scores.

The model will fit the X_train (features) and y_train (labels) data to the Perceptron Algorithm
to learn how to predict on given features.
'''
from sklearn.linear_model import Perceptron

perceptron_Model = Perceptron(n_iter_no_change=10, random_state=0)
perceptron_Model.fit(X_train,y_train)


# In[96]:
perceptron_y_pred = perceptron_Model.predict(X_test)
perceptron_precision = precision_score(y_test, perceptron_y_pred, average=None)
perceptron_recall = recall_score(y_test,perceptron_y_pred, average=None)
perceptron_f1 = f1_score(y_test,perceptron_y_pred, average=None)

perceptron_precision1 = precision_score(y_test, perceptron_y_pred, average="macro")
perceptron_recall1 = recall_score(y_test,perceptron_y_pred, average="macro")
perceptron_f11 = f1_score(y_test,perceptron_y_pred, average="macro")

for i in range(len(perceptron_precision)):
    print(f"{perceptron_precision[i]},{perceptron_recall[i]},{perceptron_f1[i]}")

print(f"{perceptron_precision1},{perceptron_recall1},{perceptron_f11}")


# # SVM

# In[76]:
'''
SVM model is from the sklearn library and uses the parameters C=1.0, random_state=0

The linearSVC model uses squared hinge loss by default. 
The C parameter is the regularization parameter.
Random_state=0 ensures that we get the same train and test sets across different executions.

The model will fit the svm_X_train (features) and svm_y_train (labels) data to the SVM Algorithm
to learn how to predict on given features.
'''

from sklearn import svm


# SVM_model = svm.SVC(C=1.0, kernel='linear', degree=1, gamma='auto')
SVM_model = svm.LinearSVC(C=0.1, random_state=0)
SVM_model.fit(X_train, y_train)


# In[77]:
SVM_y_pred = SVM_model.predict(X_test)
SVM_precision = precision_score(y_test, SVM_y_pred, average=None)
SVM_recall = recall_score(y_test, SVM_y_pred, average=None)
SVM_f1 = f1_score(y_test, SVM_y_pred, average=None)

SVM_precision1 = precision_score(y_test, SVM_y_pred, average="macro")
SVM_recall1 = recall_score(y_test, SVM_y_pred, average="macro")
SVM_f11 = f1_score(y_test, SVM_y_pred, average="macro")

for i in range(len(SVM_precision)):
    print(f"{SVM_precision[i]},{SVM_recall[i]},{SVM_f1[i]}")

print(f"{SVM_precision1},{SVM_recall1},{SVM_f11}")
# # Logistic Regression

# In[93]:
'''
Logistic Regression model is imported from the sklearn library.

For the parameter, random_state=0 , we get the same train and test sets across different executions.
The multi_class parameter is multinominal because our data is not binary and must classify 5 different classes.
The solver parameter specifies which algorithm to use in the optimization problem. I used the SAGA solver
because it is faster on larger datasets like this one and handles multinomial loss.
Max_iter means that the model is only allowed to iterate over the data 100 times before stopping.

The model will fit the X_train (features) and y_train (labels) data to the Logistic Regression Algorithm
to learn how to predict on given features.
'''

from sklearn.linear_model import LogisticRegression

logistic_Model = LogisticRegression(random_state=0, multi_class='multinomial', solver='saga', max_iter=100).fit(X_train, y_train)


# In[94]:
logistic_y_pred = logistic_Model.predict(X_test)
logisitic_precision = precision_score(y_test, logistic_y_pred, average=None)
logistic_recall = recall_score(y_test, logistic_y_pred, average=None)
logistic_f1 = f1_score(y_test, logistic_y_pred, average=None)

logisitic_precision1 = precision_score(y_test, logistic_y_pred, average="macro")
logistic_recall1 = recall_score(y_test,logistic_y_pred, average="macro")
logistic_f11 = f1_score(y_test,logistic_y_pred, average="macro")

for i in range(len(logisitic_precision)):
    print(f"{logisitic_precision[i]},{logistic_recall[i]},{logistic_f1[i]}")

print(f"{logisitic_precision1},{logistic_recall1},{logistic_f11}")


# # Naive Bayes

# In[91]:
'''
The Multinominal Naive Bayes model is imported from the sklearn library.

This classifier is suitable for classification with discrete features (e.g., word counts for text classification).

The model will fit the X_train (features) and y_train (labels) data to the Multinominal Naive Bayes Algorithm
to learn how to predict on given features.
'''
from sklearn.naive_bayes import MultinomialNB

NB_model = MultinomialNB()
NB_model.fit(X_train,y_train)


# In[92]:
NB_y_pred = NB_model.predict(X_test)
NB_precision = precision_score(y_test, NB_y_pred, average=None)
NB_recall = recall_score(y_test, NB_y_pred, average=None)
NB_f1 = f1_score(y_test, NB_y_pred, average=None)

NB_precision1 = precision_score(y_test, NB_y_pred, average="macro")
NB_recall1 = recall_score(y_test,NB_y_pred, average="macro")
NB_f11 = f1_score(y_test,NB_y_pred, average="macro")

for i in range(len(logisitic_precision)):
    print(f"{NB_precision[i]},{NB_recall[i]},{NB_f1[i]}")

print(f"{NB_precision1},{NB_recall1},{NB_f11}")


