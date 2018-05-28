import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

'''def isiNan():
	for y in values2:
		null_index = train[y].isnull()
		train.loc[~null_index, [y]] = scaler.fit_transform(train.loc[~null_index, [y]])
		train.apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x), replace=True), x))
		train.fillna(pd.Series(np.random.choice(train[y], size=len(train.index))))
		train[y] = train[y].replace(float('nan'),random.uniform(0,1))'''
def isiNan():
	train.apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x), replace=True), x))
	for y in values2:
		train[y] = train[y].replace(float('nan'),random.uniform(0,1))

pca = PCA(.9,svd_solver='full')

df = pd.read_csv("C:\\Users\\owner\\Desktop\\Semester 6\\SoftKom\\bouts_out_new.csv")
#df.index += 1

values = {"age_A"  ,"age_B"  ,"height_A"  ,"height_B"  ,"reach_A"  ,"reach_B"  ,"stance_A"  ,"stance_B"  ,"weight_A"  ,"weight_B"  ,"won_A"  ,"won_B"  ,"lost_A"  ,"lost_B"  ,"drawn_A"  ,"drawn_B"  ,"kos_A"  ,"kos_B"  ,"result"  ,"decision"  ,"judge1_A"  ,"judge1_B"  ,"judge2_A"  ,"judge2_B"  ,"judge3_A" ,"judge3_B"}
values2 = list(values)
values3 = {'PC 1' , 'PC 2'}
values2.pop(values2.index('result'))
values2.pop(values2.index('decision'))

scaler = MinMaxScaler()

df['stance_A'] = df['stance_A'].replace("orthodox",0)
df['stance_A'] = df['stance_A'].replace("southpaw",1)
df['stance_B'] = df['stance_B'].replace("orthodox",0)
df['stance_B'] = df['stance_B'].replace("southpaw",1)
df['result'] = df['result'].replace("draw",0)
df['result'] = df['result'].replace("win_A",1)
df['result'] = df['result'].replace("win_B",2)
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

target1 = df['result']
target2 = df['decision']
train = df
train = train.drop('result',1)
train = train.drop('decision',1)

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

isiNan()

trainComponent = pca.fit_transform(train)
print(pca.explained_variance_ratio_) 
kolom = []
ctr = 1
for x in pca.explained_variance_ratio_:
	kolom.append("PC-"+str(ctr))
	ctr += 1

train = pd.DataFrame(data = trainComponent,columns = kolom)

X_train, X_test, y_train, y_test = train_test_split(train,target1,test_size=0.2)
print (train)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#lm = MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter=1000,hidden_layer_sizes=(15, 5), random_state=1)
lm = MLPClassifier(hidden_layer_sizes=(100,25,5,25,100),activation='logistic', max_iter=1000, alpha=0.33,
					 solver='adam', verbose=True, random_state=21,tol=0.000000001)

model = lm.fit(X_train, y_train)
predict = model.predict(X_test)

print(accuracy_score(y_test, predict))

plt.scatter(y_test, predict)
plt.xlabel("True Values")
plt.ylabel("Predictions")
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
  