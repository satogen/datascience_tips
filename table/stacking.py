# 学習データに対する「目的変数を知らない」予測値と、テストデータに対する予測値を返す関数
def predict_cv(model, train_x, train_y, test_x, cv_f, metrics=None):
    preds = []
    preds_test = []
    va_idxes = []
    metric_values = []

    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    for i, (tr_idx, va_idx) in enumerate(cv_f):
      # データの作成
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        # モデルの学習
        model.fit(tr_x, tr_y, va_x, va_y)

        # 予測結果の格納
        pred = model.predict(va_x)
        preds.append(pred)

        # 精度検証
        if metrics:
          v = metrics(va_y, pred)
          if v:
            metric_values.append(v)

        # テストデータに対する予測
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_idxes.append(va_idx)

    # バリデーションデータに対する予測値を連結し、その後元の順序に並べ直す
    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    # テストデータに対する予測値の平均をとる
    preds_test = np.mean(preds_test, axis=0)

    if metric_values:
      print(f"CV: {np.mean(metric_values)}")

    return pred_train, preds_test