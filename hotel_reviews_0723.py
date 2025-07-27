import pandas as pd
'''
第一次使用下載
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
'''
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize


# 讀入資料
data = pd.read_csv('tripadvisor_hotel_reviews.csv')
print(data.info())
print(data['Rating'].value_counts())

# print(len(stop_words))

# 去除標點符號等非英文字及停用字
def get_clean_review(review):
    review_clean = []
    review = re.sub(r'[^a-zA-Z]',' ',review)
    review_split = review.split()
    for word in review_split:
        if word not in stop_words:
            review_clean.append(word.lower())
    return ' '.join(review_clean) # 將split list合成一個字串

# 新增clean_review欄位
data['clean_review'] = data['Review'].apply(get_clean_review)
'''
for i in range(len(data['Review'])):
    review = data.loc[i,'Review']
    data.loc[i,'clean_review'] = get_clean_review(review)
'''

# 轉換詞性
# 將 NLTK 詞性代碼 轉換成 WordNet 用的詞性代碼
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
# 轉成原形，新增lemmatized_review欄位
lemmatizer = WordNetLemmatizer()
def lemmatize_review(review):
    words = word_tokenize(review) # 斷詞
    pos_tags = pos_tag(words) # 標註 NLTK 詞性代碼
    review_lemmatized = []
    # 轉換成 WordNet 用的詞性代碼，並轉換為原形時態
    for word, pos in pos_tags:
        word_lem = lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        review_lemmatized.append(word_lem)
    return ' '.join(review_lemmatized)

data['lemmatized_review'] = data['clean_review'].apply(lemmatize_review)


# TF-IDF向量化
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(data['lemmatized_review'])

# 每個詞出現在幾筆文件中
doc_freq = (X > 0).sum(axis=0)
doc_freq = np.array(doc_freq).flatten()

# 取得詞彙名稱
terms = vectorizer.get_feature_names_out()

top_terms_df = pd.DataFrame({'term': terms,'doc_freq': doc_freq,'ratio': doc_freq / X.shape[0]})  # 出現比例

# 前 10 個最常出現的詞
# 用以決定max_df
top_10 = top_terms_df.sort_values(by='doc_freq', ascending=False).head(10)
print(top_10)

# 以特徵數量，訂定min_df
for min_df in [1, 3, 5, 10]:
    for max_df in [0.95, 0.9, 0.85]:
        vec = TfidfVectorizer(min_df=min_df, max_df=max_df)
        X = vec.fit_transform(data['lemmatized_review'])
        print(f"min_df={min_df}, max_df={max_df}, 特徵數量={X.shape[1]}")


# print 混淆矩陣及classification report

def get_matrix_and_report (model, test_x, test_y, prediction):
    print('Accuracy:', model.score(test_x, test_y))
    print()
    print('confusion matrix')
    print(pd.crosstab(test_y, prediction))
    print()
    print('classification report')
    print(classification_report(test_y, prediction))
    

# 模型1
vectorizer = TfidfVectorizer(max_features=3000,ngram_range=(1, 2),max_df=0.85,min_df=3)

tfidf_matrix = vectorizer.fit_transform(data['lemmatized_review'])
feature_names = vectorizer.get_feature_names_out()

# 將稀疏矩陣轉為普通 numpy 矩陣，再轉為 DataFrame
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# print(df_tfidf.info())
# print(df_tfidf.head(10))

# 邏輯回歸預測分數
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report

X = df_tfidf
y = np.array(data['Rating'])

logr = LogisticRegression()

Xtrain, Xtest, ytrain, ytest = tts(X, y, test_size = 0.3, random_state = 100)
logr.fit(Xtrain, ytrain)
pred_rating = logr.predict(Xtest)

print()
print('Module 1')
get_matrix_and_report(logr, Xtest, ytest, pred_rating)


# 模型2
# 建模參數加上 class_weight='balanced'
logr_bal = LogisticRegression(class_weight='balanced', multi_class='multinomial', \
                              solver='lbfgs', max_iter=1000)

Xtrain_1, Xtest_1, ytrain_1, ytest_1 = tts(X, y, test_size = 0.3, random_state = 100)
logr_bal.fit(Xtrain_1, ytrain_1)
pred_rating_1 = logr_bal.predict(Xtest_1)

print()
print('Module 2')
get_matrix_and_report(logr_bal, Xtest_1, ytest_1, pred_rating_1)


# 模型3
# 調整TF-IDF 參數 max_df 為 0.7
vectorizer_07 = TfidfVectorizer(max_features=3000,ngram_range=(1, 2),max_df=0.7,min_df=3)
tfidf_matrix_07 = vectorizer_07.fit_transform(data['lemmatized_review'])
feature_names_07 = vectorizer_07.get_feature_names_out()
df_tfidf_07 = pd.DataFrame(tfidf_matrix_07.toarray(), columns=feature_names_07)

X_07 = df_tfidf_07

logr_07 = LogisticRegression()

Xtrain_07, Xtest_07, ytrain_07, ytest_07 = tts(X_07, y, test_size = 0.3, random_state = 100)
logr_07.fit(Xtrain_07, ytrain_07)
pred_rating_07 = logr_07.predict(Xtest_07)

print()
print('Module 3')
get_matrix_and_report(logr_07, Xtest_07, ytest_07, pred_rating_07)


'''
# 無放入簡報內之模型
# 以模型三為基礎，建模參數加上 class_weight='balanced'
logr_bal_07 = LogisticRegression(class_weight='balanced', multi_class='multinomial',\
                                 solver='lbfgs', max_iter=1000)

Xtrain_1_07, Xtest_1_07, ytrain_1_07, ytest_1_07 = tts(X_07, y, test_size = 0.3, random_state = 100)
logr_bal_07.fit(Xtrain_1_07, ytrain_1_07)
pred_rating_1_07 = logr_bal_07.predict(Xtest_1_07)

print()
print('Module (no show on presentation)')
print('Accuracy:', logr_bal_07.score(Xtest_1_07, ytest_1_07))
print()
print('confusion matrix')
print(pd.crosstab(ytest_1_07, pred_rating_1_07))
print()
print('classification report')
print(classification_report(ytest_1_07, pred_rating_1_07))
'''

# 模型4
# 以目前表現較好的模型3為基礎，調整其他參數
# max_features 由3000調整為 4000
vectorizer_07_ver1 = TfidfVectorizer(max_features=4000,ngram_range=(1, 2),max_df=0.7,\
                                     min_df=3)
tfidf_matrix_07_ver1 = vectorizer_07_ver1.fit_transform(data['lemmatized_review'])

feature_names_07_ver1 = vectorizer_07_ver1.get_feature_names_out()
df_tfidf_07_ver1 = pd.DataFrame(tfidf_matrix_07_ver1.toarray(), columns=feature_names_07_ver1)

X_07_ver1 = df_tfidf_07_ver1

logr_07_ver1 = LogisticRegression()

Xtrain_07_ver1, Xtest_07_ver1, ytrain_07_ver1, ytest_07_ver1 = tts(X_07_ver1, y, \
                                                                   test_size = 0.3, \
                                                                   random_state = 100)
logr_07_ver1.fit(Xtrain_07_ver1, ytrain_07_ver1)
pred_rating_07_ver1 = logr_07_ver1.predict(Xtest_07_ver1)

print()
print('Module 4')
get_matrix_and_report(logr_07_ver1, Xtest_07_ver1, ytest_07_ver1, pred_rating_07_ver1)

# 模型5
# max_df 由0.7調整為0.55
# max_features 由3000調整為 4000
vectorizer_055 = TfidfVectorizer(max_features=4000,ngram_range=(1, 2),max_df=0.55,min_df=3)
tfidf_matrix_055 = vectorizer_055.fit_transform(data['lemmatized_review'])

feature_names_055 = vectorizer_055.get_feature_names_out()
df_tfidf_055 = pd.DataFrame(tfidf_matrix_055.toarray(), columns=feature_names_055)

X_055 = df_tfidf_055

logr_055 = LogisticRegression()

Xtrain_055, Xtest_055, ytrain_055, ytest_055 = tts(X_055, y, test_size = 0.3, random_state = 100)
logr_055.fit(Xtrain_055, ytrain_055)
pred_rating_055 = logr_055.predict(Xtest_055)

print()
print('Module 5')
get_matrix_and_report(logr_055, Xtest_055, ytest_055, pred_rating_055)

# 模型6
# 模型5為基礎，建模參數加上 class_weight='balanced'
logr_055_ver1 = LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='lbfgs', max_iter=1000)

Xtrain_055, Xtest_055, ytrain_055, ytest_055 = tts(X_055, y, test_size = 0.3, random_state = 100)
logr_055_ver1.fit(Xtrain_055, ytrain_055)
pred_rating_055_ver1 = logr_055_ver1.predict(Xtest_055)

print()
print('module 6')
get_matrix_and_report(logr_055_ver1, Xtest_055, ytest_055, pred_rating_055_ver1)

