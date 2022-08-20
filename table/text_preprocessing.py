from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
# 必要なパッケージ、データ、ドキュメントのダウンロード (時間がかかる)
nltk.download('all')
# https://www.kaggle.com/code/andradaolteanu/ii-commonlit-bert-vs-roberta-w-b-testing


############ TFIDF ############
# tf = TfidfVectorizer(df)
# _df = tf.fit_transform(df[key])
###################################
class TfidfPandas:
    def __init__(self, df):
        self.df = df
        self.tf = TfidfVectorizer()

    def fit_transform(self, key, params=None):
        values = self.tf.fit_transform(self.df[key], **params)
        return pd.DataFrame(values.toarray(), columns=self.tf.get_feature_names())

############ テキストカウント ############
# 文章内の単語のカウントをデータフレーム全体に処理する。
# 出現頻度が低い単語を削除する処理に利用する

## example
# count_word = CountWord(df)
# word_freq_pd = count_word.fit_transform(key)
# word_freq_pd["freq"].quantile(0.80)
# # 四分位数、四分位範囲
# q1=word_freq_pd['freq'].quantile(.25)
# q3=word_freq_pd['freq'].quantile(.75)
# iqr=q3-q1
# limit_high=q3+iqr*1.5
# print(limit_high)
# target_word_list = word_freq_pd[word_freq_pd['freq'] >= limit_high ]['word'].to_list()
# plt.boxplot(word_freq_pd[word_freq_pd['freq'] >= limit_high ]["freq"])

###################################

class CountWord:
    def __init__(self, df):
        self.df = df
    
    def fit_transform(self, key):
        sentences = self.df[key].to_list()
        words = []
        for sentence in sentences:
            words.extend(word_tokenize(sentence))

        word_freq_pd = pd.DataFrame(
            {'word': words}
            ).assign(freq=1).groupby('word')['freq'].count().reset_index().sort_values('freq', ascending=False)
        
        return word_freq_pd
