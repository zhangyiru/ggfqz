import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

class Processing():
    # 黑名单获取
    def _get_blacklist(self, train):
        cheat = train[train['label'] == 1]
        noCheat = train[train['label'] == 0]
        blacklist_dic = {}
        for f in ['adidmd5', 'imeimd5']:
            w_s = []
            s = set(cheat[f])
            blacklist_dic[f] = s
        return blacklist_dic

    # 特征工程
    def _feature_eng(self, data):

        # 1.屏幕面积=屏幕宽*高
        area = data['h'] * data['w']
        ss = StandardScaler()
        area = np.array(area).reshape(-1, 1)
        h = np.array(data['h']).reshape(-1, 1)
        w = np.array(data['w']).reshape(-1, 1)

        area = ss.fit_transform(area)
        h = ss.fit_transform(h)
        w = ss.fit_transform(w)
        data['area'] = area

        h = h.reshape(-1, )
        w = w.reshape(-1, )
        # print(h.shape, w.shape)
        data['h'] = h
        data['w'] = w

        # 2.orientation有异常值,2和90处理为0
        orientation = data['orientation']
        orientation[orientation == 2.0] = 0
        orientation[orientation == 90.0] = 0
        data['orientation'] = orientation

        # 3.carrier中-1赋值为0
        carrier = data['carrier']
        carrier[carrier == -1.0] = 0
        data['carrier'] = carrier

        # 4.ntt中0,7（未知）为0，1，2（宽带）为1，4,5,6（移动网络）为2，蜂窝网络未知为3
        ntt = data['ntt']
        ntt[(ntt == 7)] = 0
        ntt[(ntt == 2)] = 1
        ntt[(ntt == 4) | (ntt == 5) | (ntt == 6)] = 2
        data['ntt'] = ntt.astype('str')

        # 5.province中-1设为1，其他设为0
        # 把-1设置为1，其他设置为0
        province = data['province']
        province[province != -1] = 0
        province[province == -1] = 1
        data['province'] = province

        # 6.lan中类似中文语言代码的赋值为'zh'
        lan = data['lan']
        lan[(lan == 'zh-CN') | (lan == 'zh') | (lan == 'cn') | (lan == 'zh_CN') | (lan == 'Zh-CN') | (
                lan == 'zh-cn') | (
                    lan == 'ZH') | (lan == 'CN')] = 'zh'

        # 7.合并make
        # make
        def make_fix(x):
            """
            iphone,iPhone,Apple,APPLE>--apple
            redmi>--xiaomi
            honor>--huawei
            Best sony,Best-sony,Best_sony,BESTSONY>--best_sony
            :param x:
            :return:
            """
            x = x.lower()
            if 'iphone' in x or 'apple' in x:
                return 'apple'
            if '华为' in x or 'huawei' in x or "荣耀" in x:
                return 'huawei'
            if "魅族" in x:
                return 'meizu'
            if "金立" in x:
                return 'gionee'
            if "三星" in x:
                return 'samsung'
            if 'xiaomi' in x or 'redmi' in x:
                return 'xiaomi'
            if 'oppo' in x:
                return 'oppo'
            return x

        data['make'] = data['make'].astype('str').apply(lambda x: x.lower())
        data['make'] = data['make'].apply(make_fix)

        #8. hour time
        data['time'] = pd.to_datetime(data['nginxtime'], unit='ms')
        data['hour'] = data['time'].dt.hour.astype('str')

        #9.经纬度
        data['lon'] = pd.read_csv("data_process/city_lon.csv").astype('int')
        data['lat'] = pd.read_csv("data_process/city_lat.csv").astype('int')

        #10.apptype
        data['apptype'] = data['apptype'].astype('str')

        #train_test = pd.concat([train, test], ignore_index=True, sort=True)
        data['label'] = data['label'].fillna(-1).astype(int)
        #train_test['time'] = pd.to_datetime(train_test['nginxtime'], unit='ms')
        #train_test['hour'] = train_test['time'].dt.hour.astype('str')
        data.fillna('null_value', inplace=True)

        new_test = data[data['label'] == -1]
        new_train = data[data['label'] != -1]
        return new_train, new_test


class TrainModels():
    # 黑名单作弊判断
    def _judge_black(self, blacklist_dic, test):
        judge_cheat_sid = set()
        judge_features = list(blacklist_dic.keys())
        judge_df = test[judge_features + ['sid']]
        judge_df['label'] = [0] * len(judge_df)
        for f in judge_features:
            s = blacklist_dic[f]
            judge_df['label'] = judge_df.apply(lambda x: 1 if (x[f] in s or x['label'] == 1) else 0, axis=1)
        return judge_df[['sid', 'label']]

    # 利用catboost做二分类
    def _judge_catboost(self, train, test, categorical_features_indices, features):
        model = CatBoostClassifier(
            iterations=950,
            depth=8,
            cat_features=categorical_features_indices,
            learning_rate=0.1,
            custom_metric='F1',
            eval_metric='F1',
            random_seed=2019,
            l2_leaf_reg=80,
            #logging_level='Silent',
            task_type='GPU',
            gpu_ram_part=0.3,
            devices='1'
        )
        model.fit(train[features], train['label'])

        fea_ = model.feature_importances_
        fea_name = model.feature_names_
        fea = pd.Series(fea_)
        fea.rename("rank", inplace=True)
        fea_name = pd.Series(fea_name)
        fea_name.rename("name", inplace=True)
        feature_importance = pd.concat([fea, fea_name], axis=1)
        feature_importance.sort_values(by='rank',inplace=True)

        feature_importance.to_csv("feature_importance.csv",index=0)

        y_pred = model.predict(test[features]).tolist()

        judge_df = pd.DataFrame()
        judge_df['sid'] = test['sid'].tolist()
        judge_df['label'] = y_pred
        judge_df['label'] = judge_df['label'].apply(lambda x: 1 if x >= 0.5 else 0)
        return judge_df[['sid', 'label']]


if __name__ == "__main__":
    data = pd.read_csv("data.csv")

    #train = data[:1000000]
    #test = data[1000000:]
    proce_module = Processing()
    model_module = TrainModels()
    # 黑名单
    #blacklist_dic = proce_module._get_blacklist(train)
    #judge_by_blackList = model_module._judge_black(blacklist_dic, test)
    #judge_by_blackList.to_csv('judge_by_blackList.csv', index=False, encoding='utf-8')
    # 二分类---使用catboost
    features = ['pkgname', 'ver', 'adunitshowid', 'mediashowid', 'apptype', 'adidmd5', 'imeimd5', 'ip', 'macmd5',
                'openudidmd5',
                'reqrealip', 'city', 'province', 'idfamd5', 'dvctype', 'model', 'make', 'ntt',
                'carrier', 'os', 'osv', 'orientation', 'lan', 'h', 'w', 'ppi', 'area','hour','lon','lat']

    new_train, new_test = proce_module._feature_eng(data)
    print('ok')
    print(features)
    print("-------------")
    categorical_features_indices = np.where(data[features].dtypes != np.float)[0]
    print(categorical_features_indices)



    judge_by_catboost = model_module._judge_catboost(new_train, new_test, categorical_features_indices,features)


    #judge_by_catboost.to_csv('judge_by_catboost.csv', index=False, encoding='utf-8')
