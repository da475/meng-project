
  Usage Guidelines for the Support-Vector-Machine (SVM) Code
==============================================================


* The Project Files that are a part of the SVM Code are as follows :
-> preprocess_svm.py
-> slack_svm.py
-> spiralling
-> spiralling.c
-> spiralling.csv
-> svm_model.py
-> svm_tester.py


* Most of the Code is directly usable Python Scripts hence there is no build script.

* slack_svm.py and svm_model.py are the most important of the files.
  They are used in conjuction for creating an SVM model with given set of HyperParameters
  and calculating the efficacy of the trained model for a given DataSet.

* The Code is guaranteed to work on any Operating System and Platform
  as long as standard Python 2 or above is available.

* The only external dependancies are :
-> The NumPy Library for Python
-> The CSV Library for Python


* The Command-Line for using the SVM Code :
  $ python svm_model.py <learning_rate> <slack_coeff> <convergence_thresh>

* The process returns a value which is the error for the SVM Model created with the
  supplied HyperParameters learning_rate, slack_coeff and convergence_thresh.

* The Data required by the model is assumed to be present in 2 .npy files in the same directory
  The filenames are expected to be processed_for_svm_data.npy and processed_for_svm_labels.npy

