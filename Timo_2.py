import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import matplotlib as mpl
import math
import _pickle as cPickle
'''def isiNan():
	for y in values2:
		null_index = train[y].isnull()
		train.loc[~null_index, [y]] = scaler.fit_transform(train.loc[~null_index, [y]])
		train.apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x), replace=True), x))
		train.fillna(pd.Series(np.random.choice(train[y], size=len(train.index))))
		train[y] = train[y].replace(float('nan'),random.uniform(0,1))
'''
def dataClean(train):
	train2 = train.drop('result',1)
	train2 = train2.drop('decision',1)
	for x in train2:
		train = train[~np.isnan(train[x])]
	return train

def pcaAwal(train):
	trainComponent = pca.fit_transform(train)
	print(pca.explained_variance_ratio_) 
	kolom = []
	ctr = 1
	for x in pca.explained_variance_ratio_:
		kolom.append("PC-"+str(ctr))
		ctr += 1

	train = pd.DataFrame(data = trainComponent,columns = kolom)
	return train,ctr
def gantijadibinertarget1():
	for x in y_train:
		if x==0.0:
			cobaY.append([1,0,0])
		elif x==1.0:
			cobaY.append([0,1,0])
		else:
			cobaY.append([0,0,1])

def gantijadibinertarget2():
	for x in y_train:
		if x==0:
			cobaY.append([1,0,0,0,0,0,0,0,0,0])
		elif x==1:
			cobaY.append([0,1,0,0,0,0,0,0,0,0])
		elif x==2:
			cobaY.append([0,0,1,0,0,0,0,0,0,0])
		elif x==3:
			cobaY.append([0,0,0,1,0,0,0,0,0,0])
		elif x==4:
			cobaY.append([0,0,0,0,1,0,0,0,0,0])
		elif x==5:
			cobaY.append([0,0,0,0,0,1,0,0,0,0])
		elif x==6:
			cobaY.append([0,0,0,0,0,0,1,0,0,0])
		elif x==7:
			cobaY.append([0,0,0,0,0,0,0,1,0,0])
		elif x==8:
			cobaY.append([0,0,0,0,0,0,0,0,1,0])
		elif x==9:
			cobaY.append([0,0,0,0,0,0,0,0,0,1])
				
def baliktarget1():
	ctr = 0
	for x in predict:
		#print(x,target1[ctr])
		ctr+=1
		if (x == [1,0,0]).all():
			predict2.append(0.0)
		elif (x == [0,1,0]).all():
			predict2.append(1.0)
		else:
			predict2.append(2.0)

def baliktarget2():
	ctr = 0
	for x in predict:
		#print(x,target1[ctr])
		ctr+=1
		if (x == [1,0,0,0,0,0,0,0,0,0]).all():
			predict2.append(0)
		elif (x == [0,1,0,0,0,0,0,0,0,0]).all():
			predict2.append(1)
		elif (x == [0,0,1,0,0,0,0,0,0,0]).all():
			predict2.append(2)
		elif (x == [0,0,0,1,0,0,0,0,0,0]).all():
			predict2.append(3)
		elif (x == [0,0,0,0,1,0,0,0,0,0]).all():
			predict2.append(4)
		elif (x == [0,0,0,0,0,1,0,0,0,0]).all():
			predict2.append(5)
		elif (x == [0,0,0,0,0,0,1,0,0,0]).all():
			predict2.append(6)
		elif (x == [0,0,0,0,0,0,0,1,0,0]).all():
			predict2.append(7)
		elif (x == [0,0,0,0,0,0,0,0,1,0]).all():
			predict2.append(8)
		else:
			predict2.append(9)

def isiNan(train):
	train.apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x), replace=True), x))
	for y in values2:
		diisi = [] 
		diisi = list(train[y].loc[~np.isnan(train[y])])
		diisi = diisi[:len(train[y].loc[np.isnan(train[y])])]
		diisi = diisi if len(diisi)>0 else [0]
		banyak = len(train[y].loc[np.isnan(train[y])])/len(diisi) if ((len(train[y].loc[np.isnan(train[y])])/len(diisi)) > 0) else 1.0
		diisi = np.repeat(np.array(diisi),math.ceil(banyak))
		diisi = diisi[:len(train[y].loc[np.isnan(train[y])])]
		random.shuffle(diisi)
		#print(train[y].loc[np.isnan(train[y])])
		train[y].loc[np.isnan(train[y])] = train[y].loc[np.isnan(train[y])] if len(diisi) <= 1 else diisi
		#train.loc[~np.isnan(train[y]), [y]] = scaler.fit_transform(train.loc[~np.isnan(train[y]), [y]])

def arrHiddenPakai():
	arrHidden = []
	for x in range(1,ctr):
		arrHidden.append(ctr-x)
		if((ctr-x)<4):
			break
	for x in range(4,ctr):
		arrHidden.append(x)
	#print(cobaY)
	print(arrHidden)
pca = PCA(.99,svd_solver='full')

df = pd.read_csv("C:\\Users\\owner\\Desktop\\Semester 6\\Softkom\\SoftkomTimo\\bouts_out_new.csv")
#df.index += 1

values = {"age_A"  ,"age_B"  ,"height_A"  ,"height_B"  ,"reach_A"  ,"reach_B"  ,"stance_A"  ,"stance_B"  ,"weight_A"  ,"weight_B"  ,"won_A"  ,"won_B"  ,"lost_A"  ,"lost_B"  ,"drawn_A"  ,"drawn_B"  ,"kos_A"  ,"kos_B"  ,"result"  ,"decision"  ,"judge1_A"  ,"judge1_B"  ,"judge2_A"  ,"judge2_B"  ,"judge3_A" ,"judge3_B"}
values2 = list(values)

values3 = {'PC 1' , 'PC 2'}
values2.pop(values2.index('result'))
values2.pop(values2.index('decision'))
values2.pop(values2.index("judge1_A"))
values2.pop(values2.index("judge1_B"))
values2.pop(values2.index("judge2_A"))
values2.pop(values2.index("judge2_B"))
values2.pop(values2.index("judge3_A"))
values2.pop(values2.index("judge3_B"))

#scaler = MinMaxScaler()
scaler = StandardScaler()

df['stance_A'] = df['stance_A'].replace("orthodox",0)
df['stance_A'] = df['stance_A'].replace("southpaw",1)
df['stance_B'] = df['stance_B'].replace("orthodox",0)
df['stance_B'] = df['stance_B'].replace("southpaw",1)
df['result'] = df['result'].replace("draw",0.0)
df['result'] = df['result'].replace("win_A",1.0)
df['result'] = df['result'].replace("win_B",2.0)
df['decision'] = df['decision'].replace("DQ",0)
df['decision'] = df['decision'].replace("KO",1)
df['decision'] = df['decision'].replace("MD",2)
df['decision'] = df['decision'].replace("NWS",3)
df['decision'] = df['decision'].replace("PTS",4)
df['decision'] = df['decision'].replace("RTD",5)
df['decision'] = df['decision'].replace("SD",6)
df['decision'] = df['decision'].replace("TD",7)
df['decision'] = df['decision'].replace("TKO",8)
df['decision'] = df['decision'].replace("UD",9)

'''
isiNan()
trainComponent = pca.fit_transform(train)
train = pd.DataFrame(data = trainComponent,columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([train, target1], axis = 1)
#print(train)
print(finalDf)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
targets = ['draw', 'win_A', 'win_B']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
	indicesToKeep = finalDf['result'] == target
	ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
			   , finalDf.loc[indicesToKeep, 'principal component 2']
			   , c = color
			   , s = 50)
ax.legend(targets)
ax.grid()
fig.show()
plt.show()'''

#isiNan(train)
#train = train.replace(float('nan'),0)
#print(train)

maxim=0
train = df
train = dataClean(train)
target1 = train['result']
target2 = train['decision']
train = train.drop('result',1)
train = train.drop('decision',1)
#train = train.drop("judge1_A",1)
#train = train.drop("judge1_B",1)
#train = train.drop("judge2_A",1)
#train = train.drop("judge2_B",1)
#train = train.drop("judge3_A",1)
#train = train.drop("judge3_B",1)

#isiNan(train)
#train = scaler.fit_transform(train)
train,ctr = pcaAwal(train)
print(train)
X_train, X_test, y_train, y_test = train_test_split(train,target1,test_size=0.2,random_state=40)
print(train)
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)
cobaY = []
gantijadibinertarget1()
X_train = scaler.fit_transform(X_train)

#lm = MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=1000,hidden_layer_sizes=(15, 5), random_state=1)
#lm = MLPClassifier(hidden_layer_sizes=(100,95,90,85,80,75,70,65,60,55,50,45,40,35,30,25,20,15,10,5,10,15,20,25,30,35,40,45,50,55,60,65,70,76,80,85,90,95,100),activation='relu', max_iter=1000, alpha=0.0001,
#					 solver='adam', verbose=True, random_state=1,tol=0.000000001)
lm = MLPClassifier(hidden_layer_sizes=(15,10,5,10,),activation='relu', max_iter=10000, alpha=0.00001,
					 solver='adam', verbose=True, random_state=1,tol=0.000001,warm_start=True)
#with open('relu_classifier.pkl', 'rb') as fid:
#    model = cPickle.load(fid)
model = lm.fit(X_train, y_train)
for epoch in range(15000):
	print("Iter - ",epoch)
	print("--------------")
	model = model.partial_fit(X_train, y_train)
predict = model.predict(X_test.values.tolist())
with open('relu_classifier_pca099.pkl', 'wb') as fid:
    cPickle.dump(model, fid)
#print(predict)
#print(accuracy_score(y_test.values.tolist(), predict))
#print(y_test.values.tolist())
#print(predict)
predict2 = []
baliktarget1()
print(accuracy_score(y_test.values.tolist(), predict))
'''plt.scatter(X_test['PC-1'],y_test,label='True Values',color='red')
plt.scatter(X_test['PC-1'],predict,label='Predict',color='blue')
plt.scatter(y_test,predict,label='True Values vs Predicted Values',color='green')
line_up, = plt.plot(y_test, label='True Values')
line_down, = plt.plot(predict, label='Predict')'''
plt.scatter(y_test, predict)
plt.xlabel('True Values')
plt.ylabel('Prediction')
'''xkutrain = [[35,27,179,175,178,179,0,0,160,160,37,49,0,1,0,1,33,34],
			[25,30,166,170,170,170,1,1,125,125,38,42,2,2,1,0,29,33],
			[26,31,175,185,179,185,0,0,164,164,48,50,1,2,1,1,34,32],
			[25,29,175,174,179,180,0,0,155,155,46,31,1,3,1,0,32,19],
			[28,26,176,175,180,179,0,0,154,154,23,47,0,1,1,1,13,33],
			[27,22,177,175,183,179,1,1,155,153,26,41,0,0,0,1,14,30]]
ykutrain = [0,0,1,1,2,2]
model = lm.fit(xkutrain,ykutrain)
xkupred = [[29,30,160,160,163,161,0,0,114,113,46,42,0,4,0,1,38,39],[15,23,175,176,179,178,0,0,155,155,4,3,0,5,0,0,3,1]]
ykupred = [[2],[0]]
predict = model.predict(xkupred)
print(predict)
plt.scatter(ykupred,predict)'''
plt.show()
#print(train.to_string())
'''for y in values:
	null_index = df[y].isnull()
	df.loc[~null_index, [y]] = scaler.fit_transform(df.loc[~null_index, [y]])
	df.apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x), replace=True), x))
	df.fillna(pd.Series(np.random.choice(df[y], size=len(df.index))))
	df[y] = df[y].replace(float('nan'),random.uniform(0,1))
print(df.to_string())'''
#print(df.to_string())
#print(scaler.fit(df))
  