
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import BernoulliNB


# In[2]:



con = sqlite3.connect('database.sqlite') 


# In[3]:


filtered_data=pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3
""",con)
filtered_data


# In[4]:


def partition(x):
    if x < 3:
        return 'negative'
    return 'positive'

#changing reviews with score less than 3 to be positive and vice-versa
actualScore = filtered_data['Score']
positiveNegative = actualScore.map(partition) 
filtered_data['Score'] = positiveNegative


# In[5]:


sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')


# In[6]:


final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
final.shape


# In[7]:


display= pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3 AND Id=44737 OR Id=64422
ORDER BY ProductID
""", con)


# In[8]:


final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]


# In[9]:


data=pd.DataFrame(final)


# In[10]:


sampled_data=data.sample(frac=1)


# In[11]:


final=pd.DataFrame(sampled_data)


# In[12]:


final_data=final[0:1000]


# In[13]:


final_data=final_data.sort_values('Time')


# In[14]:


count_vect = CountVectorizer(ngram_range=(1,2)) #in scikit-learn
final_counts = count_vect.fit_transform(final_data['Text'].values)


# In[15]:


import seaborn as sn
from sklearn.manifold import TSNE


# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english')) #set of stopwords
sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer

def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned
print(stop)
print('************************************')
print(sno.stem('tasty'))


# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import cross_validation


# In[18]:


y = np.array(final_data['Score'])


# In[19]:


from sklearn.preprocessing import StandardScaler
standardised_data=StandardScaler(with_mean= False ).fit_transform(final_counts)


# In[20]:


X_train, X_test, y_train, y_test = cross_validation.train_test_split(standardised_data, y, test_size=0.2, random_state=0)
myList = list(range(0,50))
neighbors = list(filter(lambda x: x % 2 != 0, myList))
cv_scores = []
# perform 10-fold cross validation
for k in neighbors:
    clf = BernoulliNB(alpha=k, binarize=0.0, fit_prior=True, class_prior=None)
    scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_alpha = neighbors[MSE.index(min(MSE))]
print('\nThe optimal alpha is %d.' % optimal_alpha)

# plot misclassification error vs k 
plt.plot(neighbors, MSE)

for xy in zip(neighbors, np.round(MSE,3)):
    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

print("the misclassification error for each k value is : ", np.round(MSE,3))


# In[21]:


clf_optimal = BernoulliNB(alpha=optimal_alpha , binarize=0.0 , fit_prior=False, class_prior=[1,1])

# fitting the model
clf_optimal.fit(X_train, y_train)

# predict the response
pred = clf_optimal.predict(X_test)

# evaluate accuracy
acc = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the BernoulliNB classifier for k = %d is %f%%' % (optimal_alpha, acc))


# In[22]:


from sklearn.metrics import confusion_matrix
clf_optimal = BernoulliNB(alpha=optimal_alpha , binarize=0.0 , fit_prior=False, class_prior=[1,1])

# fitting the model
clf_optimal.fit(X_train, y_train)
print(clf_optimal.coef_)
# predict the response
pred = clf_optimal.predict(X_test)

# evaluate accuracy
matrix = confusion_matrix(y_test, pred) 
tn,fp,fn,tp= confusion_matrix(y_test, pred).ravel()
print(tn,fp,fn,tp)
precision=tp/(tp+fp)
recall=tp/(fn+tp)
f1=(2*((precision*recall)/(precision+recall)))
print("recall is:",recall)
print("precision is:",precision)
print("f1 score is:",f1)

print(matrix)


# In[24]:


#showing how output will come
pred = clf_optimal.predict(X_test[0])
print(pred)

