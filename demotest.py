import numpy as np                  # 存储大型矩阵 数组
import pandas as pd                 # 基于np的数据结构化工具集
import matplotlib.pyplot as plt      # 绘图数据可视化 基于numpy
plt.rcParams["font.sans-serif"] = "SimHei"  # 解决中文乱码问题
import seaborn as sns    # 在plt上核心库上进行更高阶封装
from sklearn import metrics # 评价指标函数名称
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split        # 函数划分训练集 测试集
from sklearn.linear_model import LogisticRegression         # LR回归
from sklearn.metrics import accuracy_score       # 分类准确率分数
from sklearn import model_selection

df_train1 = pd.read_csv(r'd:\pythondata\train_format1.csv')      # 读取训练数据  已确定prob
df_test = pd.read_csv(r'd:\pythondata\test_format1.csv')        # 读取测试数据  未确定prob
user_info = pd.read_csv(r'd:\pythondata\user_info_format1.csv')   # 读取用户个人信息  包括用户id 年龄范围 性别
user_log = pd.read_csv(r'd:\pythondata\user_log_format1.csv')    # 读取用户操作日志  交互信息
print(df_test.shape,df_train1.shape)       # 测试数据的维数 261477✖3   训练数据的维数 260864✖3
print(user_info.shape,user_log.shape)     # 用户个人信息维数 424170✖3  用户交互信息维数 54925330✖7 （列）

# 数据读取压缩占用内存大小
# def compressData(inputData):
#     '''
#     parameters: inputData: pd.Dataframe
#     return: inputData: pd.Dataframe
#     Purpose: compress input data and resave according to type
#     '''
#     for eachType in set(inputData.dtypes.values):
#         if 'int' in str(eachType):
#             for i in inputData.select_dtypes(eachType).columns.values:
#                 if inputData[i].min() < 0:
#                     inputData[i] = pd.to_numeric(inputData[i], downcast='signed')
#                 else:
#                     inputData[i] = pd.to_numeric(inputData[i], downcast='unsigned')
#         elif 'float' in str(eachType):
#             for i in inputData.select_dtypes(eachType).columns.values:
#                 inputData[i] = pd.to_numeric(inputData[i], downcast='float')
#         elif 'object' in str(eachType):
#             for i in inputData.select_dtypes(eachType).columns.values:
#                 inputData[i] = inputData[i].astype('category')
#     return inputData

print('训练集中，因变量 cls 分类情况：')
print(df_train1['label'].agg(['value_counts']).T)
print('='*55 + '\n')

ytrain = df_train1['label']

Xtrain = df_train1.loc[:,:]


# sos = SMOTE(random_state=0,sampling_strategy='auto')
# X_sos, y_sos = sos.fit_resample(Xtrain, ytrain)

rus = RandomUnderSampler(sampling_strategy={0:26320 }, random_state=0)
df_train, y_rus = rus.fit_resample(Xtrain, ytrain)


user_info.info()      # 用户个人信息显示每列的名称及类型

print(user_info.head(10))    # 打印前10行

user_info['age_range'].replace(0.0,np.nan,inplace=True)  # 把年龄为0.0的替换为nan
user_info['gender'].replace(2.0,np.nan,inplace=True)     # 把性别2。0的替换为nan
user_info.info()           # 再次查看用户个人信息 一些为0.0或2.0的变成nan  个数也变少了

user_info['age_range'].replace(np.nan,-1,inplace=True)   # 把年龄为nan的 替换为-1   不创建新的对象
user_info['gender'].replace(np.nan,-1,inplace=True)      # 把性别为nan的替换为-1    不创建新的对象

fig = plt.figure(figsize = (10, 6))                      # 指定宽高（英寸）的图画
x = np.array(["NULL","<18","18-24","25-29","30-34","35-39","40-49",">=50"])  # 横轴为年龄的分布
                                               #<18岁为1；[18,24]为2； [25,29]为3； [30,34]为4；[35,39]为5；[40,49]为6； > = 50时为7和8
y = np.array([user_info[user_info['age_range'] == -1]['age_range'].count(),     # 统计各年龄段的人数
             user_info[user_info['age_range'] == 1]['age_range'].count(),
             user_info[user_info['age_range'] == 2]['age_range'].count(),
             user_info[user_info['age_range'] == 3]['age_range'].count(),
             user_info[user_info['age_range'] == 4]['age_range'].count(),
             user_info[user_info['age_range'] == 5]['age_range'].count(),
             user_info[user_info['age_range'] == 6]['age_range'].count(),
             user_info[user_info['age_range'] == 7]['age_range'].count() + user_info[user_info['age_range'] == 8]['age_range'].count()])
plt.bar(x,y,label='人数')        # 条形图bar
plt.legend()                    # 表示不同类型的文本标签图案 loc位置
plt.show()
plt.title('用户年龄分布')

sns.countplot(x = 'age_range', order = [-1,1,2,3,4,5,6,7,8], data = user_info)   # 按年龄分布 横轴为-1到8 数据来源用户个人信息
plt.show()
plt.title('用户性别分布')

sns.countplot(x='gender',order = [-1,0,1],data = user_info)    # 按性别分布 横轴-1 0 1
plt.show()
plt.title('用户性别分布')

sns.countplot(x = 'age_range', order = [-1,1,2,3,4,5,6,7,8],hue= 'gender',data = user_info)    #  总体按年龄划分 其中再以性别划分
plt.show()
plt.title('用户性别年龄分布')
# 年纪的缺省值不少，性别的缺省值倒是不多。
# 用户年纪主要分布在18-34岁，且主要为女性。
# 缺失值处理的话，先简单处理一下，把缺失值都做删除处理吧，后面继续尝试的话可以试试填充缺失值
# 后来又注释掉了，没有删，因为这里是原始数据，应该在建立好特征之后再删吧

user_info['age_range'].replace(-1,np.nan,inplace=True)    # 再把原来的替换回来nan
user_info['gender'].replace(-1,np.nan,inplace=True)
#user_info = user_info.dropna()
#user_info.info()
sns.countplot(x = 'age_range', order = [-1,1,2,3,4,5,6,7,8],hue= 'gender',data = user_info)     #  再来统计年龄性别
plt.show()
plt.title('用户性别年龄分布')

print(user_log.head())

user_log.isnull().sum(axis=0)     # 统计缺失值 按行来统计

#user_log = user_log.dropna()
user_log.isnull().sum(axis=0)

user_log.info()          # 显示用户日志列名及类型

print(df_train.head(10))

df_train.info()     # 显示训练数据的列名及类型等信息

user_log['time_stamp'].hist(bins = 9)   # hist 绘制直方图，bins为直方图柱数
plt.show()

sns.countplot(x = 'action_type', order = [0,1,2,3],data = user_log) # 用户日志中 操作类型统计
plt.show()

df_train[df_train['label'] == 1]   #label（1位重复买家）列值都为1

print(user_log[(user_log['user_id'] == 34176) & (user_log['seller_id'] == 3906)])  # 查看指定用户id和商家id的数据行

df_train.head()     # 查看前5行

user_log.head()

df_train = pd.merge(df_train,user_info,on="user_id",how="left")     # 合并数据集 基于用户id 并都放在左边的数据集里   特征年龄和性别的添加
df_train.head()

total_logs_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"]]).count().reset_index()[["user_id","seller_id","item_id"]]
# 用户日志按 用户id和商户id 统计  并重置索引为用户id 商户id 物品id
total_logs_temp.head(10)

total_logs_temp.rename(columns={"seller_id":"merchant_id","item_id":"total_logs"},inplace=True)     # 重命名
total_logs_temp.head()

df_train = pd.merge(df_train,total_logs_temp,on=["user_id","merchant_id"],how="left")    # 再根据前两个来添加特征 特征为购买的物品id
df_train.head()

unique_item_ids_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["item_id"]]).count().reset_index()[["user_id","seller_id","item_id"]]
# 用户日志 再按个人id 商户id 物品id 重设索引
unique_item_ids_temp.head(10)

unique_item_ids_temp1 = unique_item_ids_temp.groupby([unique_item_ids_temp["user_id"],unique_item_ids_temp["seller_id"]]).count().reset_index()
# 添加特征用户 在每个商店买的不同商品的个数
unique_item_ids_temp1.head(10)

unique_item_ids_temp1.rename(columns={"seller_id":"merchant_id","item_id":"unique_item_ids"},inplace=True)
# 重命名
unique_item_ids_temp1.head(10)

df_train = pd.merge(df_train,unique_item_ids_temp1,on=["user_id","merchant_id"],how="left")
# 将特征添加至训练集
df_train.head()

categories_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["cat_id"]]).count().reset_index()[["user_id","seller_id","cat_id"]]
# 种类特征添加 先统计在各个商家卖的种类类别
categories_temp.head(20)

categories_temp1 = categories_temp.groupby([categories_temp["user_id"],categories_temp["seller_id"]]).count().reset_index()
# cat_id重新计数索引
categories_temp1.head(10)

categories_temp1.rename(columns={"seller_id":"merchant_id","cat_id":"categories"},inplace=True)
# 重命名
categories_temp1.head(10)

df_train = pd.merge(df_train,categories_temp1,on=["user_id","merchant_id"],how="left")
# 将特征添加至训练集
df_train.head(10)

browse_days_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["time_stamp"]]).count().reset_index()[["user_id","seller_id","time_stamp"]]
# 统计按个人id 商户id 购买时间 统计
browse_days_temp.head(10)

browse_days_temp1 = browse_days_temp.groupby([browse_days_temp["user_id"],browse_days_temp["seller_id"]]).count().reset_index()
# 重设 购买时间 计数索引（有购买两次的）
browse_days_temp1.head(10)

browse_days_temp1.rename(columns={"seller_id":"merchant_id","time_stamp":"browse_days"},inplace=True)
# 重命名
browse_days_temp1.head(10)

df_train = pd.merge(df_train,browse_days_temp1,on=["user_id","merchant_id"],how="left")
# 将 购买时间特征 加入训练集
df_train.head(10)

one_clicks_temp = user_log.groupby([user_log["user_id"],user_log["seller_id"],user_log["action_type"]]).count().reset_index()[["user_id","seller_id","action_type","item_id"]]
# 按 用户id 商家id 操作类型 计数  并索引带有操作商品id的个数 （重复） 总数
one_clicks_temp.head(10)

one_clicks_temp.rename(columns={"seller_id":"merchant_id","item_id":"times"},inplace=True)
# 重命名
one_clicks_temp.head(10)

one_clicks_temp["one_clicks"] = one_clicks_temp["action_type"] == 0
one_clicks_temp["one_clicks"] = one_clicks_temp["one_clicks"] * one_clicks_temp["times"]
# 统计 点击次数
one_clicks_temp.head(10)

one_clicks_temp["shopping_carts"] = one_clicks_temp["action_type"] == 1
one_clicks_temp["shopping_carts"] = one_clicks_temp["shopping_carts"] * one_clicks_temp["times"]
# 统计 添加购物车次数
one_clicks_temp.head(10)

one_clicks_temp["purchase_times"] = one_clicks_temp["action_type"] == 2
one_clicks_temp["purchase_times"] = one_clicks_temp["purchase_times"] * one_clicks_temp["times"]
# 统计 购买次数
one_clicks_temp.head(10)

one_clicks_temp["favourite_times"] = one_clicks_temp["action_type"] == 3
one_clicks_temp["favourite_times"] = one_clicks_temp["favourite_times"] * one_clicks_temp["times"]
# 统计 加入购物车次数
one_clicks_temp.head(10)

four_features = one_clicks_temp.groupby([one_clicks_temp["user_id"],one_clicks_temp["merchant_id"]]).sum().reset_index()
# 将相同 用户id 和商家id 合并 统计
four_features.head(10)

four_features = four_features.drop(["action_type","times"], axis=1)  # 将 操作类型扔去 和 总次数扔去
df_train = pd.merge(df_train,four_features,on=["user_id","merchant_id"],how="left")   # 将 四个 特征 加入训练集
df_train.head(10)

# 处理缺失值
df_train.info()

df_train.isnull().sum(axis=0)    # 查看各个缺失值的总数

df_train = df_train.fillna(method='ffill')   # ffill向前填充
# 缺失值向前填充
df_train.info()

plt.style.use('ggplot')   # 使用自带的样式美化
sns.countplot(x = 'age_range', order = [1,2,3,4,5,6,7,8],hue= 'gender',data = df_train)
plt.show()
plt.title('训练集用户性别年龄分布')

colnm = df_train.columns.tolist()
print(colnm)    # 打印所有特征
plt.figure(figsize = (5, 4))
color = sns.color_palette()   # 调色盘

df_train[colnm[5]].hist(range=[0,80],bins = 80,color = color[1])   # 横轴为第五列total_logs 直方图 范围0到80   柱体80个 颜色1号色
plt.xlabel(colnm[5],fontsize = 12)
plt.ylabel('用户数')
plt.show()

df_train[colnm[6]].hist(range=[0,40],bins = 40,color = color[1])    # 横轴为第六列 unique_item_ids
plt.xlabel(colnm[6],fontsize = 12)
plt.ylabel('用户数')
plt.show()

df_train[colnm[7]].hist(range=[0,10],bins = 10,color = color[1])    # 横轴为第七列 categories
plt.xlabel(colnm[7],fontsize = 12)
plt.ylabel('用户数')
plt.show()

df_train[colnm[8]].hist(range=[0,10],bins = 10,color = color[1])    # 横轴为第八列 browse_days
plt.xlabel(colnm[8],fontsize = 12)
plt.ylabel('用户数')
plt.show()

df_train[colnm[9]].hist(range=[0,50],bins = 50,color = color[1])    # 横轴为第九列 one_clicks
plt.xlabel(colnm[9],fontsize = 12)
plt.ylabel('用户单击次数统计')
plt.show()

df_train[colnm[10]].hist(range=[0,3],bins = 3,color = color[1])     # 横轴为第十列 shopping_carts
plt.xlabel(colnm[10],fontsize = 12)
plt.ylabel('用户数')
plt.show()

df_train[colnm[11]].hist(range=[0,6],bins = 7,color = color[1])     # 横轴为第十一列 purchase_times
plt.xlabel(colnm[11],fontsize = 12)
plt.ylabel("用户数")

df_train[colnm[12]].hist(range=[0,6],bins = 6,color = color[1])     # 横轴为第十二列 favourite_times
plt.xlabel(colnm[12],fontsize = 12)
plt.ylabel("用户数")

sns.set_style("dark")        #设置主题风格  制作热力图

plt.figure(figsize = (10,8))
colnm = df_train.columns.tolist()[2:13]           # 训练数据的需要训练的特征
mcorr = df_train[colnm].corr()              # 返回计算相关系数的表格
# np.zero_like的意思就是生成一个和你所给数组a相同shape的全0数组。
mask = np.zeros_like(mcorr, dtype=np.bool)
# np.triu_indices_from()返回方阵的上三角矩阵的索引
mask[np.triu_indices_from(mask)] = True
# diverging_palette 创建分散颜色 起始颜色220 终止颜色10    true返回matplotlib colormap 不是一个颜色列表
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# 热力图 mcorr相关系是 mask突显数据 cmap指定colormap对象   square热力图为单元格为正方形  annot在单元格上显示数值  fmt数据显示格式
g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True,fmt='0.2f')
plt.show()
# 相关性好像不大，可是日志里确实也没啥可以用的其他特征了啊

Y = df_train['label']
# axis表示删除列
X = df_train.drop(['user_id','merchant_id','label'],axis = 1)
X.head(10)

Y.head(10)
# 划分训练集测试集，test_size测试数据占的比例
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25,stratify = Y,random_state = 10)
# liblinear内部使用了坐标轴下降法来迭代优化损失函数。
Logit = LogisticRegression(solver='liblinear')
Logit.fit(X_train, y_train)     # 拟合训练
Predict = Logit.predict(X_test)     # 预测测试集
Predict_proba = Logit.predict_proba(X_test)     # n行k列的数组，第i行第j列上的数值是模型预测第i个预测样本的标签为j的概率
print(Predict[0:20])
print(Predict_proba[:])
Score = accuracy_score(y_test, Predict)
Score
# 一般的准确率验证方法

print("lr.coef_: {}".format(Logit.coef_))       # 斜率      sigmoid函数中
print("lr.intercept_: {}".format(Logit.intercept_))  # 截距
# 截距与斜率

# 初始化逻辑回归算法
LogRegAlg=LogisticRegression(random_state=1,solver='liblinear')
re = LogRegAlg.fit(X,Y)
# 使用sklearn库里面的交叉验证函数获取预测准确率分数 前面为模型（对象或分类器） 特征 标签 几折验证
scores = model_selection.cross_val_score(LogRegAlg,X,Y,cv=3)
# 使用交叉验证分数的平均值作为最终的准确率
print("逻辑斯蒂回归三折准确率:")
print(scores)
print("准确率为:{:.5f} ".format(scores.mean()),'auc:%.5f' %metrics.roc_auc_score(y_test, Predict))

# 默认criterion='gini'
from sklearn.tree import DecisionTreeClassifier         # 决策树
tree = DecisionTreeClassifier(max_depth=4,random_state=0)    # 推荐深度5-20  越深越容易过拟合
tree.fit(X_train, y_train)
Predict_proba = tree.predict_proba(X_test)
Predict = tree.predict(X_test)
scoretree = model_selection.cross_val_score(tree,X_train,y_train,cv=3)
print(Predict_proba[:])

print("Accuracy on training set: {:.5f}".format(tree.score(X_train, y_train)))
print("决策树三折交叉验证:")
print(scoretree)
print("Accuracy mean on training set: {:.5f}".format(scoretree.mean()))
print("Accuracy on test set: {:.5f}".format(tree.score(X_test, y_test)),'auc:%.5f' %metrics.roc_auc_score(y_test, Predict))

from sklearn.tree import export_graphviz          #  impurity 是否显示每个节点的纯度:gini或熵    filled不同类不同颜色
export_graphviz(tree, out_file="tree.dot", class_names=["0","1"], feature_names=X.columns.tolist(), impurity=True, filled=True)
# 我们可以利用 tree 模块的 export_graphviz 函数来将树可视化。这个函数会生成一 个 .dot 格式的文件，这是一种用于保存图形的文本文件格式。 DOT语言是一种文本图形描述语言
# 设置为结点添加颜色 的选项，颜色表示每个结点中的多数类别，同时传入类别名称和特征名称，这样可以对 树正确标记
# samples：表示该节点中的样本个数，
# values：表示该节点中每个类别有多少样本，
# class：表示该节点中大多数的类别（values多的就是class的类别）。
import graphviz     # 可视化图形工具
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
# 特征重要性 在决策树或以决策树为基础的模型中
print("Feature importances:\n{}".format(tree.feature_importances_))
# 直方图 center水平对齐
plt.barh(X.columns.tolist(),height=0.5,width=tree.feature_importances_,align="center")
plt.show()

# 随机森林分类器
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=10, random_state=2)    # 10个决策树进行表决
forest.fit(X_train, y_train)
Predict_proba = forest.predict_proba(X_test)
Predict = forest.predict(X_test)
scorerfc=model_selection.cross_val_score(forest,X_train,y_train,cv=3)
print(Predict_proba[:])
print(scorerfc)

print("Accuracy on training set: {:.5f}".format(forest.score(X_train, y_train)))
print("随机森林三折交叉验证:")
print(scorerfc)
print("Accuracy mean on training set: {:.5f}".format(scorerfc.mean()))
print("Accuracy on test set: {:.5f}".format(forest.score(X_test, y_test)),'auc:%.5f' %metrics.roc_auc_score(y_test, Predict))

plt.barh(X.columns.tolist(),height=0.5,width=forest.feature_importances_,align="center")
plt.show()


# 原始数据预处理之缩放
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = X_train[X_train.columns.tolist()].astype(float)
X_test = X_test[X_test.columns.tolist()].astype(float)
scaler.fit(X_train)     # 用于计算训练数据的均值和方差， 后面就会用均值和方差来转换训练数据

# 变换数据
X_train_scaled = scaler.transform(X_train)      # 基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正太分布，使其标准化
X_test_scaled = scaler.transform(X_test)        # 测试数据必须用同一个 scaler 标准化


df_test.head()
# 添加特征
df_test = pd.merge(df_test,user_info,on="user_id",how="left")
df_test = pd.merge(df_test,total_logs_temp,on=["user_id","merchant_id"],how="left")
df_test = pd.merge(df_test,unique_item_ids_temp1,on=["user_id","merchant_id"],how="left")
df_test = pd.merge(df_test,categories_temp1,on=["user_id","merchant_id"],how="left")
df_test = pd.merge(df_test,browse_days_temp1,on=["user_id","merchant_id"],how="left")
df_test = pd.merge(df_test,four_features,on=["user_id","merchant_id"],how="left")
df_test = df_test.fillna(method='bfill')    # 纵向向前填充 默认axis=0 纵向
df_test = df_test.fillna(method='ffill')    # 缺失值向后填充
df_test.head(10)

df_test.isnull().sum(axis=0)
# 去除该三列 横向
X1 = df_test.drop(['user_id','merchant_id','prob'],axis = 1)
X1.head(10)

Predict_proba = Logit.predict_proba(X1)
df_test["Logit_prob"] = Predict_proba[:,1]
Predict_proba[0:10]

df_test.head(10)

Predict_proba = tree.predict_proba(X1)
df_test["Tree_prob"] = Predict_proba[:,1]
Predict_proba[0:10]

df_test.head(10)

Predict_proba = forest.predict_proba(X1)
df_test["Forest_prob"] = Predict_proba[:,1]
Predict_proba[0:10]



choose = ["user_id","merchant_id","Logit_prob","Tree_prob","Forest_prob"]
res = df_test[choose]
res.rename(columns={"mlp_prob":"prob"},inplace=True)
print(res.head(10))
res.to_csv(path_or_buf = r"d:\pythondata\prediction13.csv",index = False)



