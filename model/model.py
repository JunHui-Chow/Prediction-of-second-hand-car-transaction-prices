import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_core.python.keras.callbacks import LearningRateScheduler
import lightgbm as lgbm
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
import warnings

warnings.filterwarnings('ignore')


def final_model(x_train_tree, x_test_tree, y_train_tree, x_train_nn, x_test_nn, y_train_nn):
    # lgbm模型
    predictions_lgbm, oof_lgbm = lgbm_model(x_train_tree, x_test_tree, y_train_tree)
    # catb模型
    predictions_catb, oof_catb = catb_model(x_train_tree, x_test_tree, y_train_tree)
    # 树模型stack
    predictions_tree, oof_tree = stack_model(predictions_lgbm, predictions_catb, oof_lgbm, oof_catb, y_train_tree)
    # nn模型
    predictions_nn, oof_nn = nn_model()
    
    # 模型融合
    predictions = predictions_lgbm * 0.1 + predictions_catb *0.9
    oof = oof_lgbm * 0.1 + oof_catb *0.9
    point = mean_absolute_error(oof, np.expm1(y_train_tree))
    print("final model mae:{:<8.8f}".format(point))
    return predictions


def lgbm_model(x_train, x_test, y_train):
    param = {'boosting_type': 'gbdt',
             'num_leaves': 31,
             'max_depth': -1,
             'lambda_l2': 2,
             'min_data_in_leaf': 20,
             'objective': 'regression_l1',
             'learning_rate': 0.02,
             'min_child_samples': 20,
             'feature_fraction': 0.8,
             'bagging_freq': 1,
             'bagging_fraction': 0.8,
             'bagging_seed': 11,
             'metric': 'mae',
             }
    folds = KFold(n_splits=5, shuffle=True, random_state=2023)
    oof_lgbm = np.zeros(len(x_train))
    predictions_lgbm = np.zeros(len(x_test))
    for fold, (train_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
        print("lgbm fold {}".format(fold + 1))
        train_data = lgbm.Dataset(x_train[train_idx], y_train[train_idx])
        val_data = lgbm.Dataset(x_train[val_idx], y_train[val_idx])

        clf = lgbm.train(param, train_data, num_boost_round=500, valid_sets=[train_data, val_data],
                         verbose_eval=300, early_stopping_rounds=300, feval=my_feval)
        oof_lgbm[val_idx] = clf.predict(x_train[val_idx], num_iteration=clf.best_iteration)
        predictions_lgbm += clf.predict(x_test, num_iteration=clf.best_iteration) / folds.n_splits
    print("lgbm mae:{:<8.8f}".format(mean_absolute_error(np.expm1(oof_lgbm), np.expm1(y_train))))

    return np.expm1(predictions_lgbm), np.expm1(oof_lgbm)


def my_feval(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_absolute_error(np.expm1(label), np.expm1(preds))

    return 'my_feval', score, False


def catb_model(x_train, x_test, y_train):
    param = {'n_estimators': 500,
             'loss_function': 'MAE',
             'eval_metric': 'MAE',
             'learning_rate': 0.02,
             'depth': 6,
             'use_best_model': True,
             'subsample': 0.6,
             'bootstrap_type': 'Bernoulli',
             'reg_lambda': 3,
             'one_hot_max_size': 2,
             }
    folds = KFold(n_splits=5, shuffle=True, random_state=2023)
    oof_catb = np.zeros(len(x_train))
    predictions_catb = np.zeros(len(x_test))

    for fold, (train_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
        print('catb fold {}'.format(fold + 1))
        k_x_train = x_train[train_idx]
        k_y_train = y_train[train_idx]
        k_x_val = x_train[val_idx]
        k_y_val = y_train[val_idx]

        model_catb = CatBoostRegressor(**param)
        model_catb.fit(k_x_train, k_y_train, eval_set=[(k_x_val, k_y_val)], verbose=300, early_stopping_rounds=300)
        oof_catb[val_idx] = model_catb.predict(k_x_val, ntree_end=model_catb.best_iteration_)
        predictions_catb += model_catb.predict(x_test, ntree_end=model_catb.best_iteration_) / folds.n_splits
    print('catb mae:{:<8.8f}'.format(mean_absolute_error(np.expm1(oof_catb), np.expm1(y_train))))

    return np.expm1(predictions_catb), np.expm1(oof_catb)

def nn_model():
    print("------------------开始神经网络训练-------------------")
    Train_NN_data = pd.read_csv('D:/Codelearn/car/output_data/train_nn.csv', sep=' ')
    Test_NN_data = pd.read_csv('D:/Codelearn/car/output_data/test_nn.csv', sep=' ')
    numerical_cols = Train_NN_data.columns
    print(numerical_cols)
    feature_cols = [col for col in numerical_cols if col not in ['price', 'SaleID']]
    ## 提前特征列，标签列构造训练样本和测试样本
    X_data = Train_NN_data[feature_cols]
    X_test = Test_NN_data[feature_cols]
    print(X_data.shape)
    print(X_test.shape)
    x = np.array(X_data)
    y = np.array(Train_NN_data['price'])
    x_test = np.array(X_test)
    print(x)
    print(x_test)
    print(y)

    kfolder = KFold(n_splits=5, shuffle=True, random_state=2023)
    oof_nn = np.zeros(len(x))
    predictions_nn = np.zeros(len(x_test))
    predictions_train_nn = np.zeros(len(x))
    kfold = kfolder.split(x, y)
    for train_index, vali_index in kfold:
        k_x_train = x[train_index]
        k_y_train = y[train_index]
        k_x_vali = x[vali_index]
        k_y_vali = y[vali_index]

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)))
        model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)))
        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)))
        model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)))
        model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.02)))

        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['mae'])

        model.fit(k_x_train, k_y_train, batch_size=1024, epochs=10, validation_data=(k_x_vali, k_y_vali))  # callbacks=callbacks,
        oof_nn[vali_index] = model.predict(k_x_vali).reshape((model.predict(k_x_vali).shape[0],))
        predictions_nn += model.predict(x_test).reshape((model.predict(x_test).shape[0],)) / kfolder.n_splits
        predictions_train_nn += model.predict(x).reshape((model.predict(x).shape[0],)) / kfolder.n_splits

    print("NN mae: {:<8.8f}".format(mean_absolute_error(oof_nn, y)))

    output_path = 'D:/Codelearn/car2/output_data/'
    # 测试集输出
    predictions = predictions_nn
    predictions[predictions < 0] = 0
    sub = pd.DataFrame()
    sub['SaleID'] = Test_NN_data.SaleID
    sub['price'] = predictions
    sub.to_csv(output_path + 'test_nn.csv', index=False)
    # 验证集输出
    oof_nn[oof_nn < 0] = 0
    sub = pd.DataFrame()
    sub['SaleID'] = Train_NN_data.SaleID
    sub['price'] = oof_nn
    sub.to_csv(output_path + 'train_nn.csv', index=False)
    print("---------------------神经网络训练结束------------------------")
    return np.expm1(predictions_nn), np.expm1(oof_nn)


def stack_model(predictions_lgbm, predictions_catb, oof_lgbm, oof_catb, y_train):
    train_stack = np.vstack([oof_lgbm, oof_catb]).transpose()
    test_stack = np.vstack([predictions_lgbm, predictions_catb]).transpose()
    folds = RepeatedKFold(n_splits=10, n_repeats=2, random_state=2023)
    oof_tree = np.zeros(train_stack.shape[0])
    predictions_tree = np.zeros(test_stack.shape[0])
    for fold, (train_idx, val_idx) in enumerate(folds.split(train_stack, y_train)):
        print('tree fold {}'.format(fold + 1))
        k_x_train = train_stack[train_idx]
        k_y_train = y_train[train_idx]
        k_x_val = train_stack[val_idx]
        model_stack = linear_model.BayesianRidge()
        model_stack.fit(k_x_train, k_y_train)
        oof_tree[val_idx] = model_stack.predict(k_x_val)
        predictions_tree += model_stack.predict(test_stack) / 20
    print('tree mae:{:<8.8f}'.format(mean_absolute_error(oof_tree, y_train)))

    return np.expm1(predictions_tree), np.expm1(oof_tree)

