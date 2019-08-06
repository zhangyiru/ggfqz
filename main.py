import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from model_code import GBDT
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from model_code import LightGbm,XGBoost,CatBoost
from scipy import sparse
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
# 特征工程
def _feature_eng(data):

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
    data['orientation'] = orientation.astype('str')

    # 3.carrier中-1赋值为0
    carrier = data['carrier']
    carrier[carrier == -1.0] = 0
    data['carrier'] = carrier.astype('str')

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
    data['province'] = province.astype('str')

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
    lon = pd.read_csv("data_process/city_lon.csv")
    lat= pd.read_csv("data_process/city_lat.csv")
    lon.name='lon'
    lat.name='lat'

    data['lon'] = lon
    data['lon'].fillna(-1,inplace=True)
    data['lon'] = data['lon'].astype('int')
    data['lat'] = lat
    data['lat'].fillna(-1,inplace=True)
    data['lat'] = data['lat'].astype('int')

    #10.apptype
    data['apptype'] = data['apptype'].astype('str')
    #11.LabelEncoder
    data['ver'] = data['ver'].astype('str')
    data['city'] = data['city'].astype('str')
    data['dvctype'] = data['dvctype'].astype('str')
    data['model'] = data['model'].astype('str')
    data['make'] = data['make'].astype('str')
    data['osv'] = data['osv'].astype('str')
    data['lan'] = data['lan'].astype('str')

    #train_test = pd.concat([train, test], ignore_index=True, sort=True)
    data['label'] = data['label'].fillna(-1).astype(int)
    #train_test['time'] = pd.to_datetime(train_test['nginxtime'], unit='ms')
    #train_test['hour'] = train_test['time'].dt.hour.astype('str')
    data.fillna('null_value', inplace=True)

    new_test = data[data['label'] == -1]
    new_train = data[data['label'] != -1]
    return new_train, new_test

def lb_process(data):
    for f in cat_features:
        le = LabelEncoder()
        new_f = pd.Series(le.fit_transform(data[f]))
        new_f.name = f
        data.pop(f)
        data[f] = new_f
    return data

if __name__ == '__main__':
    """读取数据集"""
    data = pd.read_csv("data.csv")
    train_data , test_data = _feature_eng(data)

    """特征"""
    cat_features = ['pkgname', 'ver', 'adunitshowid', 'mediashowid', 'apptype', 'adidmd5', 'imeimd5', 'ip', 'macmd5',
                'openudidmd5','reqrealip', 'city', 'province', 'idfamd5', 'dvctype', 'model', 'make', 'ntt',
                'carrier', 'os', 'osv', 'orientation', 'lan', 'hour']
    num_features = ['h', 'w', 'ppi', 'area', 'lon', 'lat']
    print(type(data['lon'][0]))
    model = "cab"
    #选择模型进行训练
    if model =="lgb":
        data = pd.concat([train_data,test_data])
        data = lb_process(data)
        print(data.dtypes)
        train_data = data[:1000000]
        test_data = data[1000000:]
        features = cat_features+num_features
        print(features)
        """切分数据集"""
        X_train, X_test, y_train, y_test = train_test_split(
            train_data[features], train_data['label'],
            test_size=0.3, random_state=2019
        )
        #print(cat_features)
        #得到最优lightgbm模型
        gbm = LightGbm.model_lgb(X_train,X_test,y_train,y_test,cat_features)
        train_label = train_data['label']
        gbm.n_estimators = gbm.best_iteration_
        #对所有训练集训练
        gbm.fit(train_data[features], train_label)
        #测试得到label
        test_y = gbm.predict(test_data[features]).tolist()
        judge_by_lightgbm = pd.DataFrame()
        judge_by_lightgbm['sid'] = test_data['sid'].tolist()
        judge_by_lightgbm['label'] = test_y
        judge_by_lightgbm.to_csv('submit-{}-lgb.csv'.format(datetime.now().strftime('%m%d_%H%M%S')), sep=',', index=False)

    if model == "xgb":
        data = pd.concat([train_data, test_data])
        data = lb_process(data)
        train_data = data[:1000000]
        test_data = data[1000000:]
        features = cat_features + num_features
        """切分数据集"""
        '''X_train, X_test, y_train, y_test = train_test_split(
            train_data[features], train_data['label'],
            test_size=0.3, random_state=2019
        )'''
        model_xgb = XGBoost.model_xgb(train_data[features],train_data['label'])
        y_pred = model_xgb.predict(test_data[features]).tolist()

        judge_df = pd.DataFrame()
        judge_df['sid'] = test_data['sid'].tolist()
        judge_df['label'] = y_pred
        judge_df['label'] = judge_df['label'].apply(lambda x: 1 if x >= 0.5 else 0)
        judge_df.to_csv('submit-{}-lgb.csv'.format(datetime.now().strftime('%m%d_%H%M%S')), sep=',', index=False)

    if model == 'cab':
        features = cat_features+num_features
        model_cab = CatBoost.model_cab(train_data[features],train_data['label'],features)
        y_pred = model_cab.predict(test_data[features]).tolist()

        judge_df = pd.DataFrame()
        judge_df['sid'] = test_data['sid'].tolist()
        judge_df['label'] = y_pred
        judge_df['label'] = judge_df['label'].apply(lambda x: 1 if x >= 0.5 else 0)
        judge_df.to_csv('submit-{}-lgb.csv'.format(datetime.now().strftime('%m%d_%H%M%S')), sep=',', index=False)

    if model == 'gbdt':
        train = pd.read_csv("embedding_vector/train_features.csv")
        train_label = pd.read_csv("embedding_vector/train_label.csv")
        test = pd.read_csv("embedding_vector/test_feature.csv")
        #org_train = pd.read_table("round1_iflyad_anticheat_traindata.txt")
        org_test = pd.read_table("round1_iflyad_anticheat_testdata_feature.txt")
        test_sid = org_test['sid']

        model_gbdt = GBDT.model_gbdt(train,train_label)
        y_pred = model_gbdt.predict(test).tolist()

        judge_df = pd.DataFrame()
        judge_df['sid'] = test_sid.tolist()
        judge_df['label'] = y_pred
        judge_df['label'] = judge_df['label'].apply(lambda x: 1 if x >= 0.5 else 0)
        judge_df.to_csv('submit-{}-gbdt.csv'.format(datetime.now().strftime('%m%d_%H%M%S')), sep=',', index=False)
    # Pred = False
    # """验证过程"""
    # if Pred == True:
    #     y_pred = GBDT.model_gbdt(X_train,X_test,y_train)
    #     f1_macro = precision_score(y_test,y_pred)
    #     print(f1_macro)
    # else:
    #     print("test")
    #     #y_pred = XGB.model_xgb(X_train,test_data,y_train)
    #     y_pred = GBDT.model_gbdt(X_train,test_data,y_train)
    #     # #test_label为array类型
    #     test_label = pd.Series(y_pred)
    #     test_label.rename("label", inplace=True)
    #     print(test_data.shape, test_label.shape)
    #     # test_label = pd.Series({'label':test_label})
    #     test_result = pd.concat([test_sid, test_label], axis=1)
    #     test_result.to_csv("test_result.csv", index=False)


    #


