from models import ModelLightGBM, ModelCatBoost, ModelXgboost
from stacking import predict_cv
from sklearn.model_selection import KFold

NUM_ROUND = 100
VERBOSE_EVAL = -1
CLASSES = train_y.nunique()
FOLD = 5
kf = KFold(n_splits=FOLD, shuffle=True, random_state=42)

def metrics_log_loss(y, y_pred):
   # 評価関数　タスクによって変更
   score = log_loss(y, y_pred)
   print(f'Metrics: {score}')


# データの読み込み
multiclass_df = sns.load_dataset('iris')
train, test = train_test_split(multiclass_df, test_size=0.3, random_state=0)
train_X = train.drop(["target","species"], axis=1)
test_X = test.drop(["target","species"], axis=1)
train_y = train["target"]
test_y = test["target"]

# lightGBMの学習
light_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': CLASSES,
    'verbose': -1
}
model_light = ModelLightGBM(light_params, NUM_ROUND)
pred_train_light, pred_test_light = predict_cv(model_light, train_X, train_y, test_X, kf, metrics_log_loss)

# Catboostの学習
cat_params = {
    'loss_function': 'MultiClass',
    'iterations': NUM_ROUND,   
    'verbose': 0, # catboostでは学習経過を出したくない場合は0を指定
     'early_stopping_rounds':10
}
model_cat = ModelCatBoost(cat_params)
pred_train_cat, pred_test_cat = predict_cv(model_cat, train_X, train_y, test_X, kf, metrics_log_loss)

# XGboostの学習
xgb_params = {'objective': 'multi:softmax', 
          'num_class': CLASSES,
          'verbosity': 0, 'random_state': 71
}
model_xgb = ModelXgboost(xgb_params)
pred_train_xgb, pred_test_xgb = predict_cv(model_xgb, train_X, train_y, test_X, kf)