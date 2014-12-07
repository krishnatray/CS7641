This code execute one of six machine learning algorithms to detect barcodes in provided image sets.  It requires as input the image set, a binary image set of labels and the selection of the algorithm.  Performance is measured by reserving one of the images for testing and using the remaining for training.  One training image is also evaluated to assess the difference between training and test sets.  Usage is as follows:

app_name -i image_name_template -l label_name_template -a algorithm

image_name_template and label_name_template use c-style formatting to indicate image numbers. For example 'image%03d.jpg' will open a sequence of images like image001.jpg, image002.jpg ...
Options for the alogithm include: 
    'dtree' - decision tree
    'adaboost' - adaptive boost with decision trees
    'svm' - support vector machine
    'bayes' - naive bayes
    'ann' - artificial neural network
    'knn' - k nearest neighbor
