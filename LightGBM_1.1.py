import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score,roc_auc_score
from xgboost.sklearn import XGBClassifier
data_all = pd.read_csv('G:\\2020summer\\Project\\XGboost\\2.csv')
import time
start=time.clock()
features = [x for x in data_all.columns if x not in ['Result']]
X = data_all[features]
y = data_all['Result']

print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)

forest=RandomForestClassifier(n_estimators=100,random_state=2) #　随机森林
forest.fit(X_train,y_train)
forest_y_score=forest.predict_proba(X_test)
# print(forest_y_score[:,1])
forest_score=forest.score(X_test,y_test) #准确率
# print('forest_score:',forest_score)
'ranfor_score:0.7820602662929222'

Gbdt=GradientBoostingClassifier(random_state=2) #CBDT
Gbdt.fit(X_train,y_train)
Gbdt_score=Gbdt.score(X_train,y_train) #准确率
# print('Gbdt_score:',Gbdt_score)
'Gbdt_score:0.8623384430417794'

Xgbc=XGBClassifier(random_state=2)  #Xgbc
Xgbc.fit(X_train,y_train)
y_xgbc_pred=Xgbc.predict(X_test)
Xgbc_score=accuracy_score(y_test,y_xgbc_pred) #准确率
# print('Xgbc_score:',Xgbc_score)
'Xgbc_score:0.7855641205325858'

gbm=lgb.LGBMClassifier(random_state=2018)  #lgb
gbm.fit(X_train,y_train)
y_gbm_pred=gbm.predict(X_test)
gbm_score=accuracy_score(y_test,y_gbm_pred)  #准确率
print('gbm_score:',gbm_score)
'gbm_score:0.7701471618780659'



y_test_hot = label_binarize(y_test,classes =(0, 1)) # 将测试集标签数据用二值化编码的方式转换为矩阵
Gbdt_y_score = Gbdt.decision_function(X_test) # 得到Gbdt预测的损失值
#forest_fpr,forest_tpr,forest_threasholds=metrics.roc_curve(y_test_hot.ravel(),forest_y_score[:,1].ravel()) # 计算ROC的值,forest_threasholds为阈值
#Gbdt_fpr,Gbdt_tpr,Gbdt_threasholds=metrics.roc_curve(y_test_hot.ravel(),Gbdt_y_score.ravel()) # 计算ROC的值,Gbdt_threasholds为阈值

end=time.clock()
print('Running time: %s Seconds'%(end-start))

