import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
from scipy.stats import norm, skew 

# 获取数据
dataframe_train = pd.read_csv('data\BostonHousing_train.csv')
dataframe_test = pd.read_csv('data\BostonHousing_test.csv')

#数据检查
def datacheck():
    # print(dataframe_train.info())  # 检查数据集
    # print(dataframe_train['medv'].describe())  # 检查dev

    # sns.set()  # 设置seaborn默认格式
    # sns.displot(dataframe_train['medv'])  # 绘制直方图
    # plt.show()  # 绘制直方图``
    
    # print("Skewness: %.2f" %dataframe_train['medv'].skew())  # 偏度
    # print("Kurtosis: %.2f" %dataframe_train['medv'].kurt())  # 峰度

    corrmat = dataframe_train.corr()
    # print(corrmat['medv'].sort_values(ascending=False))   
    # 相关系数矩阵

    # sns.heatmap(corrmat, vmax=.8, square=True,annot=True)
    # vmax=.8表示设置最大值为0.8；square表示将Axes横纵比设置为相等，annot=True表示在方格中写入数据
    # plt.show()
       
    cols =pd.Index(['medv']).append(corrmat['medv'].sort_values(key=lambda x:abs(x),ascending=False)[1:5].index) #选择范围
    sns.pairplot(dataframe_train[cols], height = 2.5)
    plt.show()
    
#查看分布
def normality():
    # 绘制目标特征的概率分布图
    plt.figure(figsize=(16, 8))
    sns.distplot(dataframe_train['medv'], fit=norm) # 拟合标准正态分布
    plt.legend(['Normal dist'], loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')
    plt.show()
    
    plt.figure(figsize=(16, 8))
    res = stats.probplot(dataframe_train['medv'], plot=plt)
    plt.show()
    
    (mu, sigma) = norm.fit(dataframe_train['medv'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))    
    
#数据集划分     
x_train = dataframe_train.drop(['medv'], axis=1)
y_train = dataframe_train['medv'].values
x_test = dataframe_test.drop(['medv'], axis=1)
y_test = dataframe_test['medv'].values 

#正态化
if_norm = False
def normalize():
    global if_norm,y_train
    if_norm = True
    dataframe_train["medv"] = np.log1p(dataframe_train["medv"])
    y_train = dataframe_train['medv'].values


# 评估模型 
from sklearn.metrics import mean_squared_error,r2_score   
from sklearn.model_selection import RandomizedSearchCV
def evaluate(model): 
    model.fit(x_train, y_train)
    if if_norm:   
        pred = np.expm1(model.predict(x_test)) # exp(x)-1 使预测值正常化
    else:
        pred = model.predict(x_test)
        
    rmse = mean_squared_error(y_test, pred, squared=False) 
    return (r2_score(y_true=y_test,y_pred=pred),rmse)



    
#尝试不同模型  
from sklearn.linear_model import LinearRegression,ElasticNet, Lasso, BayesianRidge, LassoLarsIC #线性回归模型
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor #集成模型
from sklearn.kernel_ridge import KernelRidge #核岭回归

from sklearn.pipeline import make_pipeline #pipeline
from sklearn.preprocessing import RobustScaler #鲁棒缩放

le=LinearRegression()
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
bayesian_ridge = make_pipeline(RobustScaler(),BayesianRidge())
lasso_lars_ic = make_pipeline(RobustScaler(),LassoLarsIC())
random_forest = make_pipeline(RobustScaler(),RandomForestRegressor())
gradient_boosting = make_pipeline(RobustScaler(),GradientBoostingRegressor(alpha=0.9, learning_rate=0.05, n_estimators=300))
KRR = make_pipeline(RobustScaler(),KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5))

#挨个测试
models_name = ['le','Lasso', 'ElasticNet','bayesian_ridge','lasso_lars_ic','random_forest','gradient_boosting','KRR']
models = [le,lasso, ENet, bayesian_ridge, lasso_lars_ic, random_forest, gradient_boosting, KRR]
def test():
    for i,model in enumerate(models):
        result = evaluate(model)
        print(f'\n{models_name[i]}: \nR2_Score: {result[0]}   \nRSME: {result[1]}')


#参数调整
from sklearn.metrics import make_scorer
def adjust(model):
    param_dist = {
    'alpha': [i/10 for i in range(11)],
    'l1_ratio': [0.2, 0.4, 0.6, 0.8],
    #'degree': [2, 3, 4],
    #'coef0': [1.0, 1.5, 2.0],
    }
    
    scorer = make_scorer(mean_squared_error, squared=False)    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        scoring=scorer,
        n_iter=10,
        cv=5,
        random_state=42
    )
    random_search.fit(x_train, y_train)
    print("Best parameters:", random_search.best_params_)
    print("Best score:", random_search.best_score_)

def compare(model):
    plt.xlabel("actual")
    plt.ylabel("predict")
    plt.scatter(x=y_test, y=model.predict(x_test))
    plt.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'k--',)
    plt.show()


import joblib
def save_model(model):
    joblib.dump(model, 'model.pkl')
    
def load_model():
    model = joblib.load('model.pkl')
    return model

def main():
    model=load_model() #加载模型
    input_data = pd.read_csv("input.csv") #读取输入数据
    result = np.expm1(model.predict(input_data)) #如果对medv使用了正态化
    sub=pd.DataFrame()
    sub['pridict_medv']=result  
    sub.to_csv('output.csv',index=False)  #保存结果

if __name__=='__main__':
   main()
