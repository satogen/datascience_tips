import lightgbm as lgb
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from catboost import Pool

############ lightGBM ############
# objective 
## 下記ドキュメントのobjectiveに対応タスクと対応Lossの記述あり
## https://lightgbm.readthedocs.io/en/latest/Parameters.html
# https://mathmatical22.xyz/2020/04/12/%E3%80%90%E5%88%9D%E5%BF%83%E8%80%85%E5%90%91%E3%81%91%E3%80%91%E7%89%B9%E5%BE%B4%E9%87%8F%E9%87%8D%E8%A6%81%E5%BA%A6%E3%81%AE%E7%AE%97%E5%87%BA-lightgbm-%E3%80%90python%E3%80%91%E3%80%90%E6%A9%9F/
###################################

class ModelLightGBM:

    def __init__(self, params, nround, verbose=-1, early_stopping_rounds=10):
        self.model = None
        self.params = params
        self.num_round = nround
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X_train, y_train, X_valid, y_valid):
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid)
   
        # モデルの学習
        self.model = lgb.train(
            self.params,
            lgb_train,
            valid_sets=lgb_eval,
            num_boost_round=self.num_round,
            verbose_eval=self.verbose,
            early_stopping_rounds = self.early_stopping_rounds
        )

    def predict(self, x):
        pred = self.model.predict(x)
        return pred

    def get_feature_importance(self, x):
      importance = pd.DataFrame(
        self.model.feature_importance(), 
        index=x.columns, 
        columns=['importance'])
      return importance

############ CatBoost ############
# 指定できるハイパーパラメータはドキュメントから確認
# https://catboost.ai/en/docs/concepts/python-reference_catboost

# loss_functionで学習タスクが決定
# 指定できるタスクはこちら
# https://catboost.ai/en/docs/concepts/loss-functions
###################################

class ModelCatBoost:

    def __init__(self, params, category_features = None, text_features = None):
        self.params = params
        self.model = CatBoost(self.params)
        self.category_features = category_features
        self.text_features = text_features

    def fit(self, X_train, y_train, X_valid, y_valid):
        if self.category_features:
          ctb_train = Pool(X_train, label=y_train,cat_features=self.category_features)  
          ctb_eval  = Pool(X_valid, label=y_valid,cat_features=self.category_features)  
        elif self.text_features:
          ctb_train = Pool(X_train, label=y_train,text_features=self.text_features)  
          ctb_eval  = Pool(X_valid, label=y_valid,text_features=self.text_features)  
        else:
          ctb_train = Pool(X_train, label=y_train)  
          ctb_eval  = Pool(X_valid, label=y_valid)  

        self.model.fit(ctb_train)

    def predict(self, x):
        pred = self.model.predict(x)
        return pred

############ Xgboost ############
# 指定できるハイパーパラメータはドキュメントから確認
# https://xgboost.readthedocs.io/en/stable/parameter.html

# objectiveで学習タスクが決定
# 指定できるタスクはこちら
# https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
###################################

class ModelXgboost:

    def __init__(self, params, num_round=1000):
        self.model = None
        self.params = params
        self.num_round = num_round

    def fit(self, tr_x, tr_y, va_x, va_y):
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)

        # xgb.trainのパラメータ一覧
        # https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(self.params, dtrain, self.num_round, evals=watchlist)

    def predict(self, x):
        data = xgb.DMatrix(x)
        # 確率を出したい時は、predict_proba
        # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier.predict_proba
        pred = self.model.predict(data)
        return pred

# SVCが分類、SVRが回帰
from sklearn.svm import SVC
class ModelClassifilerSVM:

    def __init__(self, params):
        self.model = None
        self.params = params

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.model = SVC(**self.params)
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        pred = self.model.predict_proba(x)
        return pred


# 回帰のみ
from sklearn.linear_model import Ridge
class ModelRidge:

    def __init__(self, params):
        self.model = None
        self.params = params

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.model = Ridge(**self.params)
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        pred = self.model.predict(x)
        return pred


# ロジスティック回帰
# 線形モデルの分類が一般的に使われるのがロジスティックとSVM
# パラメータはこちら:https://qiita.com/FujiedaTaro/items/5784eda386146f1fd6e7

from sklearn.linear_model import LogisticRegression

class ModelLogistic:

    def __init__(self, params):
        self.model = None
        self.params = params

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.model = LogisticRegression(**self.params)
        # self.model = LogisticRegression()
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        pred = self.model.predict_proba(x)
        return pred

# KNN

from sklearn.neighbors import KNeighborsClassifier

class ModelClassifilerKNN:

    def __init__(self, params):
        self.model = None
        self.params = params

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.model = KNeighborsClassifier(**self.params)
        # self.model = KNeighborsClassifier()
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        pred = self.model.predict_proba(x)
        return pred