Despite using the WEKA dataset as instructed, we got strange results for data partitions larger than 1000 units. This was the case for both the Naive Bayes algorihtm (weka.classifiers.bayes.NaiveBayesMultinomial) and the SMO algorithm (weka.classifiers.functions.SMO). See the attatched PDF for details.

At this time we believe there is either a problem with the dataset or with the provided algorithms.

If you want to attempt this yourself, first run the following command in your terminal:

`export CLASSPATH=.:/Volumes/weka-3-8-4/weka-3-8-4/weka.jar`

Then run the algorithm with

`java weka.classifiers.bayes.NaiveBayesMultinomial -t spam train 1000.arff -T spam test.arff`
