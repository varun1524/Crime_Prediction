# Crime-Prediction
Crime Prediction

-crimepred.ipynb can be opened on jupyter notebook
-crimepred.py can be executed directly a executable file
-Since data can be fetched from website using API (which is implemented in the provided source code). For that user needs to install sodapy library using following command:
	pip install sodapy
-Also, it can be downloaded from the following link:
https://data.sfgov.org/Public-Safety/-Change-Notice-Police-Department-Incidents/tmnf-yvry

Libraries Used:
-Python2.7
-pandas
-Numpy
-sklearn
-sodapy

- Program can be executed in sequence in jupyter
- Following steps to execute program:
	-Execute everything till you reach to "Method for Classification"
	-From here there are methods developed for the ease of user to train classification model and perfirm prediction on testing data
	-predictCrimeCategory() can be executed to perform classification for specific district
		Parameters:
			-district name
			-Classifier
	-classifyAllDistricts() can be executed to perform classification for all the districts in San Francisco  
		Parameters:
			-Classifier
			-path where output needs to be stored
	-ensamble_classifiers() can be executed for city as well as district and it generates all 3 classifier predictions and from that performs ensembling in order to generate better prediction
		Parameters:
			-district name
			-city - if true classification will be performed on city level else on district level
	


