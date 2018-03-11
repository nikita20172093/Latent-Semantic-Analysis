from nltk.tokenize import RegexpTokenizer, word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy import sparse
import numpy as np
import shutil
import string
import random
import io
import sys
import os


###### move data to test####################
def trainToTest(root_dir,dest_dir):
	# root_dir = 'q2data/train'
	# dest_dir = 'q2data/test'
	folders_root_dir = os.listdir(root_dir)
	totalTestFile = 0
	count = len(folders_root_dir)
	for i in range(count):
		dirNameSource = root_dir+ "/" +folders_root_dir[i] + "/"
		allSourceFiles= os.listdir(dirNameSource)
		dirNameDest   = dest_dir+ "/" +folders_root_dir[i] + "/"
		numOfFiles    = len(allSourceFiles)
		testingCount  = int(.2*numOfFiles)
		totalTestFile += testingCount
		randomList    = random.sample(range(0,numOfFiles),testingCount)
		for j in randomList:
			src  = dirNameSource+allSourceFiles[j]
			dest = dirNameDest
			shutil.move(src, dest)
	return totalTestFile,folders_root_dir
###### move data to test enddddddddddd####################

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++""
######### at the end of program to move all files to training set
# dest_dir = 'q2data/train'
# root_dir = 'q2data/test'
def testToTrain(root_dir,dest_dir):
	folders_root_dir = os.listdir(root_dir)

	count = len(folders_root_dir)
	for i in range(count):
		dirNameSource = root_dir+ "/" +folders_root_dir[i] + "/"
		allSourceFiles= os.listdir(dirNameSource)
		dirNameDest   = dest_dir+ "/" +folders_root_dir[i] + "/"
		numOfFiles    = len(allSourceFiles)
		for j in range(numOfFiles):
			src  = dirNameSource+allSourceFiles[j]
			dest = dirNameDest
			shutil.move(src, dest)
	return

####################################################################


def tfidfFunction(FreqArray):	
	BOW      = FreqArray.tolist()
	termFreq = TfidfTransformer()
	tfidf    = termFreq.fit_transform(BOW)
	tfidf    = tfidf.toarray()
	return tfidf

def calAccuracy(tfidf,w,label):
	inc = 0
	cor = 0
	# print tfidf,w
	for i in range(len(tfidf)):
		val = np.dot(tfidf[i],w.T)
		ind = np.argmax(val)
		# print val,ind," ",label[i]
		if ind!=label[i]:
			inc+=1
		else:
			cor+=1

	print "Accuracy :   ",(cor*100.0)/(cor+inc)		
	return
#================================================

def calWeightVector(folders_root_dir,tfidf,label):
	w = np.zeros((len(folders_root_dir),len(tfidf[0])))
	for epoc in range(5):
		for i in range(len(tfidf)):
			val = np.dot(tfidf[i],w.T)
			ind = np.argmax(val)
			if ind!=label[i]:
				w[ind]      = np.subtract(w[ind],tfidf[i])
				w[label[i]] = np.add(w[label[i]],tfidf[i])

	return w


#============================================
def getGlobalDict(folders_root_dir,root_dir,stopWords):
	count = len(folders_root_dir)
	fileList = []
	label    = [] 
	ps = PorterStemmer()
	
	globalDict = {}
	index=0
	fileIndex = 0 
	for i in range(count):
		dirNameSource = root_dir+ "/" +folders_root_dir[i] + "/"
		allSourceFiles= os.listdir(dirNameSource)
		fileCount     = len(allSourceFiles)
		for j in range(fileCount):
			fileName = dirNameSource + allSourceFiles[j]
			fileList.append(fileName)
			label.append(int(folders_root_dir[i]))
			with io.open(fileName, 'r', encoding='ascii', errors='ignore') as fname:
				for line in fname:
					line = line.lower()
					token = word_tokenize(line)
					for word in token:
						if word not in stopWords and word not in string.punctuation and len(word)>=3:
							key = ps.stem(word).encode('ascii', 'ignore')
							if key not in globalDict:
								globalDict[key]=index
								index = index + 1
	return globalDict,label,fileList

#==========================================


def frameData(folders_root_dir,root_dir,test_dir,stopWords,totalTestFile):
	globalDict,label,fileList = getGlobalDict(folders_root_dir,root_dir,stopWords)

	rowNo = 0
	FreqArray = np.zeros((len(fileList),len(globalDict)+1))
	ps = PorterStemmer()
	row = 0
	for file in fileList:
		with io.open(file, 'r', encoding='ascii', errors='ignore') as fname:
				for line in fname:
					line = line.lower()
					token = word_tokenize(line)
					for word in token:
						if word not in stopWords and word not in string.punctuation and len(word)>3:
							key = ps.stem(word).encode('ascii', 'ignore')
							if key in globalDict:
								ind = globalDict[key]
								FreqArray[row][ind] = FreqArray[row][ind]+1 
		row = row + 1


	tfidf = tfidfFunction(FreqArray)
	###############################################################


	w = calWeightVector(folders_root_dir,tfidf,label)
	# print w
	print "+++++++++++++++++++++++++"
			
	calAccuracy(tfidf,w,label)
	##################testing#############################

	folders_root_dir = os.listdir(test_dir)
	# print folders_root_dir
	testFreqArray = np.zeros((totalTestFile,len(globalDict)+1))
	testLabel = []
	testList  = []
	row = 0
	count = len(folders_root_dir)
	for i in range(count):
		dirNameSource = test_dir+ "/" +folders_root_dir[i] + "/"
		allSourceFiles= os.listdir(dirNameSource)
		fileCount     = len(allSourceFiles)

		for j in range(fileCount):

			testLabel.append(int(folders_root_dir[i]))
			fileName = dirNameSource + allSourceFiles[j]
			with io.open(fileName, 'r', encoding='ascii', errors='ignore') as fname:
				testList.append(fileName)
				for line in fname:
					line = line.lower()
					token = word_tokenize(line)
					for word in token:
						if word not in stopWords and word not in string.punctuation and len(word)>3:
							key = ps.stem(word).encode('ascii', 'ignore')
							if key in globalDict:
								ind = globalDict[key]
								testFreqArray[row][ind] = testFreqArray[row][ind]+1 
			row = row + 1

	tfidf = tfidfFunction(testFreqArray)
	print "+++++++++++++++++++++++++"

	calAccuracy(tfidf,w,testLabel)
	
	return

#####################testing end########################



if __name__ == "__main__":
	trainPath = " "
	testPath = " "
	if len(sys.argv)==3:
		trainPath = sys.argv[1]
		testPath = sys.argv[2]
		# frameData(root,dest)
	else:
		print "Input format is wrong"
		sys.exit()
	stopWords = set(stopwords.words('english'))
	noOfFiles,folders_root_dir = trainToTest(trainPath,testPath)
	frameData(folders_root_dir,trainPath,testPath,stopWords,noOfFiles) 
	testToTrain(testPath,trainPath)