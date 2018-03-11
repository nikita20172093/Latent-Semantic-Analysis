from nltk.tokenize import RegexpTokenizer, word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy import sparse
import numpy as np
import shutil
import string
import random
import io
import os
import sys


def countOccurrences(arr, n, x):
	res = 0
	for i in range(n):
		if x == arr[i]:
			res += 1
	return res

#====================================================\
def frameData(root_dir,fileTest,fileLabel):
	folders_root_dir = os.listdir(root_dir)
	count = len(folders_root_dir)
	fileList = []
	label    = [] 
	ps = PorterStemmer()
	stopWords = set(stopwords.words('english'))
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

	rowNo = 0
	FreqArray = np.zeros((len(fileList),len(globalDict)+1))

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



	BOW      = FreqArray.tolist()
	termFreq = TfidfTransformer()
	tfidf    = termFreq.fit_transform(BOW)
	tfidf    = tfidf.toarray()


	print "+++++++++++++++++++++++++"


	testFreqArray = np.zeros((1,len(globalDict)+1))
	testLabel = []
	testList  = []
	row = 0

	for j in range(1):
		with io.open(fileTest, 'r', encoding='ascii', errors='ignore') as fname:
			# testList.append(fileTest)
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


	bowTest  = testFreqArray.tolist()
	termFreq = TfidfTransformer()
	tfidf2    = termFreq.fit_transform(bowTest)
	tfidfTest    = tfidf2.toarray()



	cosSimilarity = cosine_similarity(tfidf,tfidfTest)
	# print cosSimilarity


	topIndex = []
	for top in range(10):
		ind = np.argmax(cosSimilarity)
		topIndex.append(label[ind])
		cosSimilarity[ind] = (-1*sys.maxint)
	print "+++++++++++++++++++++++++"
	maxOcc= -1
	actlabel = -1
	#####################testing end########################
	for k in topIndex:
		value = countOccurrences(topIndex,len(topIndex),k)
		if maxOcc < value:
			maxOcc = value
			actlabel = k
	if int(actlabel) == int(fileLabel):
		print "Correct classified:   1"
	else:
		print "Incorrect classified:  0, \n*** Correct class is: ",actlabel

	print "+++++++++++++++++++++++++"
	return

#========================================================
#----------------------------------------------------------------

if __name__ == "__main__":
	if len(sys.argv)==4:
		root = sys.argv[1]
		file = sys.argv[2]
		clslabel = sys.argv[3]
		frameData(root,file,clslabel)
	else:
		print "Input format is wrong"