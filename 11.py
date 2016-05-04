"""
Solution to Problem 5.11 of Pattern Recognition (Stork, Hart and Duda)
---------------------------------------------------------------------
Support Vector Classifier
---------------------------------------------------------------------

"""

import numpy

from data import w1,w2,w3,w4

from sklearn import svm as svm1

#Create feature vectors
features_w1=[[0 for x in range(6)] for x in range(10)]
features_w2=[[0 for x in range(6)] for x in range(10)]
features_w3=[[0 for x in range(6)] for x in range(10)]
features_w4=[[0 for x in range(6)] for x in range(10)]

#Populate feature vectors
for i in range(10):
	#for w1
	x1=w1[i][0]
	x2=w1[i][1]
	features_w1[i][0]=1
	features_w1[i][1]=x1
	features_w1[i][2]=x2
	features_w1[i][3]=x1*x1
	features_w1[i][4]=x1*x2
	features_w1[i][5]=x2*x2

	#for w2
	x1=w2[i][0]
	x2=w2[i][1]
	features_w2[i][0]=1
	features_w2[i][1]=x1
	features_w2[i][2]=x2
	features_w2[i][3]=x1*x1
	features_w2[i][4]=x1*x2
	features_w2[i][5]=x2*x2

	#for w3
	x1=w3[i][0]
	x2=w3[i][1]
	features_w3[i][0]=1
	features_w3[i][1]=x1
	features_w3[i][2]=x2
	features_w3[i][3]=x1*x1
	features_w3[i][4]=x1*x2
	features_w3[i][5]=x2*x2
	
	#for w4
	x1=w4[i][0]
	x2=w4[i][1]
	features_w4[i][0]=1
	features_w4[i][1]=x1
	features_w4[i][2]=x2
	features_w4[i][3]=x1*x1
	features_w4[i][4]=x1*x2
	features_w4[i][5]=x2*x2


#Part 11.a
classifier=svm1.SVC()
X=numpy.array([features_w3[0],features_w4[0]])
y=numpy.array([3,4])

#train classifier
classifier.fit(X,y)

#test classifier performance
print(classifier.predict(features_w3[0]))

#part 11.b
classifier2=svm1.SVC()
X=numpy.array([features_w3[0],features_w3[1],features_w4[0],features_w4[1]])
y=numpy.array([3,3,4,4])

#train classifier
classifier2.fit(X,y)

#test classifier performance
print(classifier2.predict(features_w4[1]))

#part 11.c
featurelist=[]
label_list=[]

#Iterate through all 10 sets of points until misclassification happens
for x in range(0,10):
	#Get elements from w_3
	featurelist.append(features_w3[x])
	label_list.append(3)

	#Get elements from w_4
	featurelist.append(features_w4[x])
	label_list.append(4)

	#Create classifer and train it!
	classifier3=svm1.SVC()
	classifier3.fit(numpy.array(featurelist),numpy.array(label_list))
	
	#Test classifier for misclassification
	isWorking=True
	for y in range(0,x+1):
		if(classifier3.predict(features_w3[y])!=3):
			isWorking=False
		if(classifier3.predict(features_w4[y])!=4):
			isWorking=False	
		
	if(isWorking==False):
		#Stop at first misclassification
		print("Misclassification at set number: %d"%(x+1))		
		break
	else:
		print("Classification successful for set of first %d points!"%(x+1))
	

	





