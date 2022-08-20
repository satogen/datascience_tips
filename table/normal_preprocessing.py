## 標準化
# 線形モデルの前処理で利用
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# 訓練データでFit、その後Transform ,一緒にやりたい場合はfit_transform
scaler.fit(_df)
_s_df = pd.DataFrame(scaler.transform(_df))



## 次元圧縮
from sklearn.decomposition import TruncatedSVD
model_svd = TruncatedSVD(n_components=100)
model_svd.fit(_df)
_svd_df = pd.DataFrame(model_svd.transform(_df))

