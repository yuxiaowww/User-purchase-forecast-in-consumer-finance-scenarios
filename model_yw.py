# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:23:12 2018

@author: yuwei
"""

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn import metrics


path ='..//dataset//'

#%%
def loadData(path):
    "读取数据集"
    "训练集"
    #APP操作行为日志
    train_log = pd.read_table(path+'train_log.csv',sep='\t')
    #切分EVT_LBL
    train_log['EVT_LBL_0'] = train_log.EVT_LBL.map(lambda x:x.split('-')[0])
    train_log['EVT_LBL_1'] = train_log.EVT_LBL.map(lambda x:x.split('-')[1])
    train_log['EVT_LBL_2'] = train_log.EVT_LBL.map(lambda x:x.split('-')[2])
    #获取时间
    train_log['OCC_TIM_HOUR'] = train_log.OCC_TIM.map(lambda x :int(str(x)[11:13]) if int(str(x)[11:13])>1 else 24)
    train_log['OCC_TIM'] = train_log.OCC_TIM.map(lambda x :int(str(x)[8:10]))

#    #获取时间
#    train_log['date'] = train_log.OCC_TIM.map(lambda x :datetime.datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S'))
#    train_log['OCC_TIM'] = train_log.date.map(lambda x :x.day)
#    train_log['OCC_TIM_HOUR'] = train_log.date.map(lambda x :x.hour)
    #个人属性与信用卡消费数据
    train_agg = pd.read_table(path+'train_agg.csv',sep='\t')
    #标签数据
    train_flg = pd.read_table(path+'train_flg.csv',sep='\t')
    "测试集"
    test_log = pd.read_table(path+'test_log.csv',sep='\t')
    test_log['EVT_LBL_0'] = test_log.EVT_LBL.map(lambda x:x.split('-')[0])
    test_log['EVT_LBL_1'] = test_log.EVT_LBL.map(lambda x:x.split('-')[1])
    test_log['EVT_LBL_2'] = test_log.EVT_LBL.map(lambda x:x.split('-')[2])
    #获取时间
    test_log['OCC_TIM_HOUR'] = test_log.OCC_TIM.map(lambda x :int(str(x)[11:13]) if int(str(x)[11:13])>1 else 24)
    test_log['OCC_TIM'] = test_log.OCC_TIM.map(lambda x :int(str(x)[8:10]))
#    #获取时间
#    test_log['date'] = test_log.OCC_TIM.map(lambda x :datetime.datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S'))
#    test_log['OCC_TIM'] = test_log.date.map(lambda x :x.day)
#    test_log['OCC_TIM_HOUR'] = train_log.date.map(lambda x :x.hour) 
    test_agg = pd.read_table(path+'test_agg.csv',sep='\t')

    return train_log,train_agg,train_flg,test_log,test_agg

#%%
def genFeatureAgg(data,agg):
    "特征提取--agg表:V1-V30"
    
    #保存原始表
    ans = data.copy()

    "提取agg表特征"
    #V1到V30属性
    ans = pd.merge(ans,agg)
    #V1到V30求和
    ans['V1_to_V30'] = ans['V1']+ans['V2']+ans['V3']+ans['V4']+ans['V5']+ans['V6']+ans['V7']+ans['V8']+ans['V9']+ans['V10']+\
    ans['V11']+ans['V12']+ans['V13']+ans['V14']+ans['V15']+ans['V16']+ans['V17']+ans['V18']+ans['V19']+ans['V20']+\
    ans['V21']+ans['V22']+ans['V23']+ans['V24']+ans['V25']+ans['V26']+ans['V27']+ans['V28']+ans['V29']+ans['V30']
    #V1到V10求和
    ans['V1_to_V10'] = ans['V1']+ans['V2']+ans['V3']+ans['V4']+ans['V5']+ans['V6']+ans['V7']+ans['V8']+ans['V9']+ans['V10']
    #V11到V20求和
    ans['V10_to_V20'] = ans['V11']+ans['V12']+ans['V13']+ans['V14']+ans['V15'] + ans['V16']+ans['V17']+ans['V18']+ans['V19']+ans['V20']
    #V20到V30求和
    ans['V20_to_V30'] = ans['V21']+ans['V22']+ans['V23']+ans['V24']+ans['V25']+ ans['V26']+ans['V27']+ans['V28']+ans['V29']+ans['V30']
    #V1到V5求和
    ans['V1_to_V5'] = ans['V1']+ans['V2']+ans['V3']+ans['V4']+ans['V5']
    #V6到V10求和
    ans['V6_to_V10'] = ans['V6']+ans['V7']+ans['V8']+ans['V9']+ans['V10']
    #V11到V15求和
    ans['V11_to_V15'] = ans['V11']+ans['V12']+ans['V13']+ans['V14']+ans['V15']
    #V16到V20求和
    ans['V16_to_V20'] = ans['V16']+ans['V17']+ans['V18']+ans['V19']+ans['V20']
    #V21到V25求和
    ans['V21_to_V25'] = ans['V21']+ans['V22']+ans['V23']+ans['V24']+ans['V25']
    #V26到V30求和
    ans['V26_to_V30'] = ans['V26']+ans['V27']+ans['V28']+ans['V29']+ans['V30']
    #求除法
    ans['V1_to_V5_rate'] = ans['V1_to_V5']/ans['V1_to_V30']
    ans['V6_to_V10_rate'] = ans['V6_to_V10']/ans['V1_to_V30']
    ans['V11_to_V15_rate'] = ans['V11_to_V15']/ans['V1_to_V30']
    ans['V16_to_V20_rate'] = ans['V16_to_V20']/ans['V1_to_V30']
    ans['V21_to_V25_rate'] = ans['V21_to_V25']/ans['V1_to_V30']
    ans['V26_to_V30_rate'] = ans['V26_to_V30']/ans['V1_to_V30']

    return ans

#%%
def genFeatureLog(data,logData):
    "特征提取"
    #保存原始表
    ans = data.copy()
    log = logData.copy()
    
    #按不同的天数粒度提取特征
    for i in [31,24,21,18,14,10,7,5,4,3,2,1]:

       log = log[(log.OCC_TIM>=32-i)]
       #共计出现多少次
       log['count_'+str(i)] = log['USRID']
       feat = pd.pivot_table(log,index=['USRID'],values='count_'+str(i),aggfunc='count').reset_index()
       ans = pd.merge(ans,feat,on='USRID',how='left')
       #共交互多少个不同的LBL
       log['diff_lbl_'+str(i)] = log['USRID']
       feat = pd.pivot_table(log,index=['USRID','EVT_LBL'],values='diff_lbl_'+str(i),aggfunc='count').reset_index()
       feat['diff_lbl_'+str(i)] = feat['USRID']
       feat = pd.pivot_table(feat,index=['USRID'],values='diff_lbl_'+str(i),aggfunc='count').reset_index()
       ans = pd.merge(ans,feat,on='USRID',how='left')
       #共交互多少个不同的LBL_0
       log['diff_lbl_0_'+str(i)] = log['USRID']
       feat = pd.pivot_table(log,index=['USRID','EVT_LBL_0'],values='diff_lbl_0_'+str(i),aggfunc='count').reset_index()
       feat['diff_lbl_0_'+str(i)] = feat['USRID']
       feat = pd.pivot_table(feat,index=['USRID'],values='diff_lbl_0_'+str(i),aggfunc='count').reset_index()
       ans = pd.merge(ans,feat,on='USRID',how='left')
       #共交互多少个不同的LBL_1
       log['diff_lbl_1_'+str(i)] = log['USRID']
       feat = pd.pivot_table(log,index=['USRID','EVT_LBL_1'],values='diff_lbl_1_'+str(i),aggfunc='count').reset_index()
       feat['diff_lbl_1_'+str(i)] = feat['USRID']
       feat = pd.pivot_table(feat,index=['USRID'],values='diff_lbl_1_'+str(i),aggfunc='count').reset_index()
       ans = pd.merge(ans,feat,on='USRID',how='left')
       #共交互多少个不同的LBL_2
       log['diff_lbl_2_'+str(i)] = log['USRID']
       feat = pd.pivot_table(log,index=['USRID','EVT_LBL_2'],values='diff_lbl_2_'+str(i),aggfunc='count').reset_index()
       feat['diff_lbl_2_'+str(i)] = feat['USRID']
       feat = pd.pivot_table(feat,index=['USRID'],values='diff_lbl_2_'+str(i),aggfunc='count').reset_index()
       ans = pd.merge(ans,feat,on='USRID',how='left')
    
       #统计type为0的次数
       type_0 = log[log.TCH_TYP==0]
       type_0['type_0_count_'+str(i)] = type_0['USRID']
       feat = pd.pivot_table(type_0,index=['USRID'],values='type_0_count_'+str(i),aggfunc='count').reset_index()
       ans = pd.merge(ans,feat,on='USRID',how='left')
       del type_0
       #统计type为2的次数
       type_2 = log[log.TCH_TYP==2]
       type_2['type_2_count_'+str(i)] = type_2['USRID']
       feat = pd.pivot_table(type_2,index=['USRID'],values='type_2_count_'+str(i),aggfunc='count').reset_index()
       ans = pd.merge(ans,feat,on='USRID',how='left')
       del type_2
       
       #计算最近几个小时次数
       if i == 1:
          for j in [12,18,21,24]:
              log = log[(log.OCC_TIM_HOUR>=j)]
              #共计出现多少次
              log['count_hour'+str(i)] = log['USRID']
              feat = pd.pivot_table(log,index=['USRID'],values='count_hour'+str(i),aggfunc='count').reset_index()
              ans = pd.merge(ans,feat,on='USRID',how='left')
              #共交互多少个不同的LBL
              log['diff_lbl_hour'+str(i)] = log['USRID']
              feat = pd.pivot_table(log,index=['USRID','EVT_LBL'],values='diff_lbl_hour'+str(i),aggfunc='count').reset_index()
              feat['diff_lbl_hour'+str(i)] = feat['USRID']
              feat = pd.pivot_table(feat,index=['USRID'],values='diff_lbl_hour'+str(i),aggfunc='count').reset_index()
              ans = pd.merge(ans,feat,on='USRID',how='left')
              #共交互多少个不同的LBL_0
              log['diff_lbl_0_hour'+str(i)] = log['USRID']
              feat = pd.pivot_table(log,index=['USRID','EVT_LBL_0'],values='diff_lbl_0_hour'+str(i),aggfunc='count').reset_index()
              feat['diff_lbl_0_hour'+str(i)] = feat['USRID']
              feat = pd.pivot_table(feat,index=['USRID'],values='diff_lbl_0_hour'+str(i),aggfunc='count').reset_index()
              ans = pd.merge(ans,feat,on='USRID',how='left')
              #共交互多少个不同的LBL_1
              log['diff_lbl_1_hour'+str(i)] = log['USRID']
              feat = pd.pivot_table(log,index=['USRID','EVT_LBL_1'],values='diff_lbl_1_hour'+str(i),aggfunc='count').reset_index()
              feat['diff_lbl_1_hour'+str(i)] = feat['USRID']
              feat = pd.pivot_table(feat,index=['USRID'],values='diff_lbl_1_hour'+str(i),aggfunc='count').reset_index()
              ans = pd.merge(ans,feat,on='USRID',how='left')
              #共交互多少个不同的LBL_2
              log['diff_lbl_2_hour'+str(i)] = log['USRID']
              feat = pd.pivot_table(log,index=['USRID','EVT_LBL_2'],values='diff_lbl_2_hour'+str(i),aggfunc='count').reset_index()
              feat['diff_lbl_2_hour'+str(i)] = feat['USRID']
              feat = pd.pivot_table(feat,index=['USRID'],values='diff_lbl_2_hour'+str(i),aggfunc='count').reset_index()
              ans = pd.merge(ans,feat,on='USRID',how='left')
           
              #统计type为0的次数
              type_0 = log[log.TCH_TYP==0]
              type_0['type_0_count_hour'+str(i)] = type_0['USRID']
              feat = pd.pivot_table(type_0,index=['USRID'],values='type_0_count_hour'+str(i),aggfunc='count').reset_index()
              ans = pd.merge(ans,feat,on='USRID',how='left')
              del type_0
              #统计type为2的次数
              type_2 = log[log.TCH_TYP==2]
              type_2['type_2_count_hour'+str(i)] = type_2['USRID']
              feat = pd.pivot_table(type_2,index=['USRID'],values='type_2_count_hour'+str(i),aggfunc='count').reset_index()
              ans = pd.merge(ans,feat,on='USRID',how='left')
    return ans

    
#%%
def modelXgb(train,test):
    "xgb模型"
    train_y = train['FLAG'].values
                         
    train_x = train.drop(['USRID','FLAG'],axis=1).values
    test_x = test.drop(['USRID','FLAG'],axis=1).values        
                    
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              'eval_metric':'auc',
              'eta': 0.03,
              'max_depth': 6,  # 6
              'colsample_bytree': 0.8,#0.8
              'subsample': 0.8,
              'scale_pos_weight': 1,
              'min_child_weight': 18  # 2
              }
    # 训练
    watchlist = [(dtrain,'train')]
    bst = xgb.train(params, dtrain, num_boost_round=370,evals=watchlist)
    # 预测
    predict = bst.predict(dtest)
    test_xy = test[['USRID']]
    test_xy['RST'] = predict
    test_xy = test_xy.sort_values('RST', ascending=False)
    
    return test_xy   

#%%
def oneHot(logData,ans):
    ""
    log = logData.copy()
    log['EVT_LBL_0_oh'] = log['USRID']
    feat = pd.pivot_table(log,index=['USRID','EVT_LBL_0'],values='EVT_LBL_0_oh',aggfunc='count').reset_index()
    feat = feat.set_index(feat.columns.tolist()[0:2])
    feat = feat.unstack()
    feat = feat.reset_index()
    ans = pd.merge(ans,feat,on='USRID',how='left')
    
    log['EVT_LBL_1_oh'] = log['USRID']
    feat = pd.pivot_table(log,index=['USRID','EVT_LBL_1'],values='EVT_LBL_1_oh',aggfunc='count').reset_index()
    feat = feat.set_index(feat.columns.tolist()[0:2])
    feat = feat.unstack()
    feat = feat.reset_index()
    ans = pd.merge(ans,feat,on='USRID',how='left')
    
    log['EVT_LBL_2_oh'] = log['USRID']
    feat = pd.pivot_table(log,index=['USRID','EVT_LBL_2'],values='EVT_LBL_2_oh',aggfunc='count').reset_index()
    feat = feat.set_index(feat.columns.tolist()[0:2])
    feat = feat.unstack()
    feat = feat.reset_index()
    ans = pd.merge(ans,feat,on='USRID',how='left')

    return ans

#%%
def validate(all_train):
    "模型验证"
    train_x = all_train.drop(['USRID', 'FLAG'], axis=1).values
    train_y = all_train['FLAG'].values
    auc_list = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
    for train_index, test_index in skf.split(train_x, train_y):
            print('\n')
            print('Train: %s | test: %s' % (train_index, test_index))
            X_train, X_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
    
            pred_value = xgb_model(X_train, y_train, X_test)
            print(pred_value)
            print(y_test)
    
            pred_value = np.array(pred_value)
            pred_value = [ele + 1 for ele in pred_value]
    
            y_test = np.array(y_test)
            y_test = [ele + 1 for ele in y_test]
    
            fpr, tpr, thresholds = roc_curve(y_test, pred_value, pos_label=2)
            
            auc = metrics.auc(fpr, tpr)
            print('auc value:',auc)
            auc_list.append(auc)
    print('validate result:',np.mean(auc_list))
    
    
def xgb_model(train_set_x,train_set_y,test_set_x):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'binary:logistic',
              'eval_metric':'auc',
              'eta': 0.03,
              'max_depth': 6,  # 6
              'colsample_bytree': 0.8,#0.8
              'subsample': 0.8,
              'scale_pos_weight': 1,
              'min_child_weight': 18  # 2
              }
    dtrain = xgb.DMatrix(train_set_x, label=train_set_y)
    dvali = xgb.DMatrix(test_set_x)
    model = xgb.train(params, dtrain, num_boost_round=370)
    predict = model.predict(dvali)
    return predict

#%%

if __name__ == '__main__':
   "主函数入口"
   #获取原始数据
   train_log,train_agg,train_flg,test_log,test_agg = loadData(path)
   test_flg = test_agg[['USRID']];test_flg['FLAG']=-1
   #合并训练集和测试集
   flg = pd.concat([train_flg,test_flg],axis=0)
   agg = pd.concat([train_agg,test_agg],axis=0)
   log = pd.concat([train_log,test_log],axis=0)
   "特征提取：agg表"
   data = genFeatureAgg(flg,agg)
   "特征提取：log表"
   data = genFeatureLog(data,log)
   "特征提取：log表离散EVT_LBL"
   data = oneHot(log,data)
   "分割训练集和测试集"
   trainset = data[data['FLAG']!=-1]
   testset = data[data['FLAG']==-1]
   
   "模型验证"
#   validate(trainset)
   
   "模型训练"
   answer = modelXgb(trainset,testset)
   pd.Series(np.array(answer['RST'].values)).plot(figsize=(8, 8))
   answer.to_csv("yw.csv",index=None,sep='\t')


