//Based on work by Bytefish.de http://bytefish.de/blog/machine_learning_opencv/

#include <iostream>
#include <math.h>
#include <string>
#include <algorithm>
#include <ctime>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat svm(Mat& trainingData, Mat& trainingClasses, Mat& testData) {
	
    //see http://docs.opencv.org/modules/ml/doc/support_vector_machines.html#cvsvmparams
    float weight[2] = {1.0, 1.0};  //optionally bias the result towards the positive examples
    CvMat weights = Mat(2, 1, CV_32F, weight);
    CvSVMParams param = CvSVMParams();
	param.svm_type = CvSVM::C_SVC;  //classification problem
    param.kernel_type = CvSVM::RBF;
	param.degree = 0;
    param.gamma = 1;
    param.coef0 = 0;
    param.C = 1;
	param.nu = 0.0;
	param.p = 0.0;
    param.class_weights = &weights;
	param.term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
	param.term_crit.max_iter = 1000;
	param.term_crit.epsilon = 1e-3;
    
    //train
    CvSVM svm;
    svm.train(trainingData, trainingClasses, Mat(), Mat(), param);
    //svm.train_auto(trainingData, trainingClasses, Mat(), Mat(), param);  //this may take a long time
    
    //predict
	Mat predictions(testData.rows, 1, CV_32F);
    svm.predict(testData, predictions);
    
    predictions.convertTo(predictions, CV_8U);
    return predictions;
}

Mat mlp(Mat& trainingData, Mat& trainingClasses, Mat& testData) {
    
    const float stdevThreshold = 0.7; //number of standard deviations above the mean for a positive prediction
    
    //confert inputs to float
    Mat trainingClassesFloat;
    trainingClasses.convertTo(trainingClassesFloat, CV_32FC1);  //MLP requires float results
    
    //layers
    Mat layers = Mat(4, 1, CV_32SC1);
	layers.row(0) = Scalar(6);
	layers.row(1) = Scalar(10);
    layers.row(2) = Scalar(15);
    layers.row(3) = Scalar(1);
    CvANN_MLP mlp;
    mlp.create(layers);
    
    //parameters
    //see http://docs.opencv.org/modules/ml/doc/neural_networks.html#cvann-mlp-trainparams
	CvANN_MLP_TrainParams params = CvANN_MLP_TrainParams();
	CvTermCriteria criteria;
	criteria.max_iter = 1000;
	criteria.epsilon = 1e-4;
	criteria.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
	params.train_method = CvANN_MLP_TrainParams::RPROP;
	params.bp_dw_scale = 0.1;       // for BACKPROP
	params.bp_moment_scale = 0.1;   // for BACKPROP
    params.rp_dw0 = 0.1;            // for RPROP
    params.rp_dw_plus = 1.2;        // for RPROP
    params.rp_dw_minus = 0.5;       // for RPROP
    params.rp_dw_min = FLT_EPSILON; // for RPROP
    params.rp_dw_max = 50.;         // for RPROP
	params.term_crit = criteria;
    
	//train
	mlp.train(trainingData, trainingClassesFloat, cv::Mat(), cv::Mat(), params);
    
    //predict
	Mat predictions(testData.rows, 1, CV_32F);
    clock_t startTime = clock();
    mlp.predict(testData, predictions);
    cout << "MLP prediction time: " << (float) (clock() - startTime) / CLOCKS_PER_SEC << endl;
    

    //threshold
    Scalar predictionMean, predictionStdev;
    meanStdDev(predictions, predictionMean, predictionStdev);
    Mat predictionsBool = Mat(predictions.rows, 1, CV_8U);
    for (int i = 0; i < predictions.rows; i++) {
        predictionsBool.at<uchar>(i, 0) = predictions.at<float>(i, 0) > predictionMean[0] + stdevThreshold * predictionStdev[0] ? 1 : 0;
    }
    
    return predictionsBool;
}

Mat knn(Mat& trainingData, Mat& trainingClasses, Mat& testData) {
    
    const int K = 3;
    const int numSamples = 100;  //taking about 35ms per sample, so can't wait for a whole image
    
    //train
	CvKNearest knn(trainingData, trainingClasses, cv::Mat(), false, K);
    
    //predict
	Mat predictions(testData.rows, 1, CV_32F);
	for(int i = 0; i < numSamples; i++) {
        predictions.at<float>(i,0) = knn.find_nearest(testData.row(i), K);
    }
    
    predictions.convertTo(predictions, CV_8U);
    return predictions;
}

Mat bayes(Mat& trainingData, Mat& trainingClasses, Mat& testData) {
    
    //train
	CvNormalBayesClassifier bayes(trainingData, trainingClasses);
    
    //predict
	Mat predictions(testData.rows, 1, CV_32F);
    bayes.predict(testData, &predictions);
    
    predictions.convertTo(predictions, CV_8U);
    return predictions;
}

Mat decisionTree(const Mat& trainingData, const Mat& trainingClasses, const Mat& testData) {
    
    //see http://docs.opencv.org/modules/ml/doc/decision_trees.html#cvdtreeparams
    CvDTreeParams params;
    float priors[2] = {1., .1};  //bias the result towards the positive examples
    params.max_depth = 10;
    params.min_sample_count = 10;
    params.regression_accuracy = 0;
    params.use_surrogates = false;
    params.max_categories = 10;
    params.cv_folds = 10;
    params.use_1se_rule = false;
    params.truncate_pruned_tree = true;
    params.priors = priors;
    
    //set the var_type for float features and categorical lables
	Mat var_type(trainingData.cols + 1, 1, CV_8U);
    var_type.setTo(CV_VAR_NUMERICAL);  //features are float (0.0 to 255.0)
    var_type.at<unsigned char>(trainingData.cols, 0) = CV_VAR_CATEGORICAL;  //labels should be 0 or 1
    
    //train
    CvDTree dtree;
	dtree.train(trainingData, CV_ROW_SAMPLE, trainingClasses, Mat(), Mat(), var_type, Mat(), params);
    
    //predict
    Mat predictions(testData.rows, 1, CV_8U);
    for (int i = 0; i < testData.rows; i++) {
        predictions.at<uchar>(i, 0) = dtree.predict(testData.row(i))->value;
    }
    
    return predictions;
}

Mat adaBoost(Mat& trainingData, Mat& trainingClasses, Mat& testData) {
    
    //see http://docs.opencv.org/modules/ml/doc/decision_trees.html#cvdtreeparams
    //and http://docs.opencv.org/modules/ml/doc/boosting.html#cvboostparams-cvboostparams
    CvBoostParams params;
    
    //decision tree parameters
    float priors[2] = {1., 0.1};        //bias the result towards the positive examples
    params.min_sample_count = 10;
    params.regression_accuracy = 0;
    params.use_surrogates = false;
    params.max_categories = 10;
    params.use_1se_rule = false;
    params.truncate_pruned_tree = false;
    params.priors = priors;
    
    //boosted parameters
    params.max_depth = 10;               //this is generally low for boosted trees
    params.cv_folds = 0;                //cross validation not used for boosting
    params.boost_type = CvBoost::DISCRETE;
    params.weak_count = 10;
    params.weight_trim_rate = 0.95;
    
    //set the var_type for float features and categorical lables
    Mat var_type(trainingData.cols + 1, 1, CV_8U);
    var_type.setTo(CV_VAR_NUMERICAL);  //features are float (0.0 to 255.0)
    var_type.at<unsigned char>(trainingData.cols, 0) = CV_VAR_CATEGORICAL;  //labels should be 0 or 1
    
    //train
    CvBoost boost(trainingData, CV_ROW_SAMPLE, trainingClasses, Mat(), Mat(), var_type, Mat(), params);
    
    //predict
    Mat predictions(testData.rows, 1, CV_8U);
    for (int i = 0; i < testData.rows; i++) {
        predictions.at<uchar>(i, 0) = saturate_cast<uchar>(boost.predict(testData.row(i)));
    }
    
    return predictions;
}

int loadImages(string fileTemplate, Mat* imageArray, const int arraySize) {
    
    char fileName[1024];
    int numImages = 0;
    
    while(numImages < arraySize) {
        sprintf(fileName, fileTemplate.c_str(), numImages + 1);
        imageArray[numImages] = imread(fileName);
        if(!imageArray[numImages].data) {
            if(numImages == 0) {
                cerr <<  "Could not open image: " << fileName << endl ;
                return -1;
            }
            else {
                break;
            }
        }
        numImages++;
    }
    return numImages;
}

int writeToFile(const string &fileName, const Mat &matrix) {
    FileStorage file(fileName, cv::FileStorage::WRITE);
    file << "Mat" << matrix;
    file.release();
    return 0;
}

void displayOffsetWindow(const string &name, const Mat &image) {
    static int windowPosition = 0;
    const int offset = 25;
    const int displaySize = 1000;
    
    namedWindow(name);
    windowPosition = (windowPosition + offset) % displaySize;
    moveWindow(name, windowPosition, windowPosition);
    imshow(name, image);
    waitKey();
}

int features(const Mat* imageArray, const int imageArraySize, Mat &featureArray) {
    
    const int gaussianBlurKernelSize = 3;
    const int statKernelSize = 15;
    const int sobelKernelSize = 3;
    const int laplacianKernelSize = 1;
    const int gaborKernelSize = 31;
    const double gaborAspectRatio = .33;
    const double gaborVariance = gaborKernelSize / 6.;  // +/- 3 sigma
    const double gaborMinWavelength = 3;
    const double gaborMaxWavelength = 9;
    const int gaborNumAngles = 0;
    const int gaborNumWavelegths = 4;
    const int gaborNumImages = gaborNumAngles * gaborNumWavelegths;
    const int numFeatures = 6 + gaborNumImages;
    const bool showImages = false;
    
    Mat image, imageGray, imageHSV, image32F;
    Mat mu, mu2, sigma;
    Mat hue, value, saturation;
    Mat sobelX, sobelY, sobel, sobelDir;
    Mat LBP;
    Mat kernel;
    Mat gaborArray[gaborNumImages], gabor;
    Mat laplacian;
    
    int rows, cols;
    int totalFeatures = 0;
    int totalPixels = 0;
    
    //OpenCV requires features to be CV_32F, we will keep them CV_8U until the end
    featureArray.create(0, numFeatures, CV_32FC1);
    
    for (int img = 0; img < imageArraySize; img++) {
        rows = imageArray[img].rows;
        cols = imageArray[img].cols;
        imageArray[img].copyTo(image);
        GaussianBlur(image, image, Size(gaussianBlurKernelSize, gaussianBlurKernelSize), 0);
        cvtColor(image, imageGray, CV_BGR2GRAY);

        //HSV
        cvtColor(image, imageHSV, CV_BGR2HSV);
        hue.create(imageHSV.size(), CV_8U);
        value.create(imageHSV.size(), CV_8U);
        saturation.create(imageHSV.size(), CV_8U);
        Mat HSVArray[] = {hue, saturation, value};
        int channels[] = {0, 0, 1, 1, 2, 2};
        mixChannels(&imageHSV, 1, HSVArray, 3, channels, 3);
        if (showImages) {
            displayOffsetWindow("Hue", hue);
            displayOffsetWindow("Saturation", saturation);
            displayOffsetWindow("Value", value);
        }
        
        //Statistics
        imageGray.convertTo(image32F, CV_32F);
        blur(image32F, mu, Size(statKernelSize, statKernelSize));
        blur(image32F.mul(image32F), mu2, Size(statKernelSize, statKernelSize));
        cv::sqrt(mu2 - mu.mul(mu), sigma);
        normalize(mu, mu, 0.0, 255.0, NORM_MINMAX);
        normalize(sigma, sigma, 0.0, 255.0, NORM_MINMAX);
        mu.convertTo(mu, CV_8U);
        sigma.convertTo(sigma, CV_8U);
        if (showImages) {
            displayOffsetWindow("sigma", sigma);
        }
        
        //Soble
        Sobel(imageGray, sobelX, CV_8U, 1, 0, sobelKernelSize);
        Sobel(imageGray, sobelY, CV_8U, 0, 1, sobelKernelSize);
        addWeighted(sobelX, 0.5, sobelY, 0.5, 0, sobel);
        if (showImages) {
           displayOffsetWindow("Sobel", sobel);
        }

        //Laplacian
        Laplacian(imageGray, laplacian, CV_8U, laplacianKernelSize);
        if (showImages) {
            displayOffsetWindow("Laplacian", laplacian);
        }
        
        //Gabor
        for (int i = 0; i < gaborNumAngles; i++) {
            double  gaborAngle = i * CV_PI / gaborNumAngles;
            for (int j = 0; j < gaborNumWavelegths; j++) {
                double  gaborWavelength = gaborMinWavelength + (double) j * (gaborMaxWavelength - gaborMinWavelength) / gaborNumWavelegths;
                kernel = getGaborKernel(Size(gaborKernelSize, gaborKernelSize),
                                        gaborVariance,
                                        gaborAngle,
                                        gaborWavelength,
                                        1.0 / gaborAspectRatio,
                                        0,
                                        CV_32FC1);
                filter2D(imageGray, gabor, CV_8U, kernel);
                gabor.copyTo(gaborArray[i*gaborNumWavelegths + j]);
                
                //Display Images
                if (showImages) {
                    Mat imageToShow, kernelToShow;
                    char windowName[50];
                    sprintf(windowName, "angle: %.2f wavelength: %.2f", gaborAngle, gaborWavelength);
                    convertScaleAbs(kernel, kernelToShow, 256);
                    convertScaleAbs(gabor, imageToShow);
                    kernelToShow.copyTo(imageToShow(Rect(1, 1, kernelToShow.cols, kernelToShow.rows)));
                    rectangle(imageToShow, Rect(0, 0, kernelToShow.cols+1, kernelToShow.rows+1), 255);
                    displayOffsetWindow(windowName, imageToShow);

                }
            }
        }
        
        //Assign features to featureArray; values are 0.0 to 255.0
        featureArray.resize(featureArray.rows + rows * cols, featureArray.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float features[numFeatures];  // need to use temp array to force a copy (not by reference)
                
                features[0] = (float)hue.at<unsigned char>(i,j);
                features[1] = (float)saturation.at<unsigned char>(i,j);
                features[2] = (float)value.at<unsigned char>(i,j);
                features[3] = (float)sigma.at<unsigned char>(i,j);
                features[4] = (float)sobel.at<unsigned char>(i,j);
                features[5] = (float)laplacian.at<unsigned char>(i,j);
                for (int g = 0; g < gaborNumImages; g++) {
                    features[6+g] = (float)gaborArray[g].at<unsigned char>(i,j);
                }
                
                for (int k = 0; k < numFeatures; k++) {
                    featureArray.at<float>(totalPixels + i*cols + j, k) = features[k];
                }
            }
        }
        totalPixels += rows * cols;
        totalFeatures += numFeatures * totalPixels;
    }
    
    return totalFeatures;
}

int labels(const Mat* labelImageArray, const int imageArraySize, Mat &labelArray) {
    Mat imageGray;
    int rows, cols;
    int totalLabels = 0;
    
    labelArray.create(1, &totalLabels, CV_32SC1);  //type per OpenCV convention for classification

    for (int img = 0; img < imageArraySize; img++) {
        rows = labelImageArray[img].rows;
        cols = labelImageArray[img].cols;
        labelArray.resize(labelArray.rows + rows * cols, 1);
        cvtColor(labelImageArray[img], imageGray, CV_BGR2GRAY);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                labelArray.at<int>(totalLabels + i*cols + j, 0) = imageGray.at<uchar>(i,j) > 127 ? 1 : 0;
            }
        }
        totalLabels += rows * cols;
    }
    return totalLabels;
}

void randomFeatures(const Mat& srcFeatures, const Mat& srcLabels, Mat &dstFeatures, Mat &dstLables, int numFeatures) {
    
    RNG random(time(NULL));
    int row;
    
    dstFeatures.create(numFeatures, srcFeatures.cols, srcFeatures.type());
    dstLables.create(numFeatures, srcLabels.cols, srcLabels.type());
    
    for (int i = 0; i < numFeatures; i++) {
        row = random.uniform(0, srcFeatures.rows);
        srcFeatures.row(row).copyTo(dstFeatures.row(i));
        srcLabels.row(row).copyTo(dstLables.row(i));
    }
}

void evaluate(Mat& predicted, Mat& actual, string name) {
    assert(predicted.rows == actual.rows);
    int tp = 0;
    int tn = 0;
    int fp = 0;
    int fn = 0;
    int pos = 0;
    int neg = 0;
    for (int i = 0; i < actual.rows; i++) {
        int p = predicted.at<uchar>(i,0);
        int a = actual.at<uchar>(i,0);
        if (p && a)
            tp++;
        else if (!p && !a)
            tn++;
        else if (!p && a)
            fn++;
        else
            fp++;
        if (a)
            pos++;
        else
            neg++;
    }
    float tpr = (float) tp / pos;
    float fpr = (float) fp / neg;
    float accuracy = (float) (tp + tn) / (tp + tn + fp + fn);
    float precision = (float) tp / (tp + fp);
    float recall = (float) tp / (tp + fn);
    float F = 2.0 * (precision * recall) / (precision + recall);
    cout << name << endl;
    cout << "  Prior: " << (float) pos / (pos + neg) << endl;
    cout << "  True Positive Rate: " << tpr << " False Positive Rate: " << fpr << endl;
    cout << "  Accuracy: " << accuracy << " Recall: " << recall << " Precision: " << precision << " F1: " << F << endl;
}

void displayResult(const Mat* imageArray, const int index, const int numTestImages, const Mat& prediction, const char* title) {
    int start = 0;
    int end = 0;
    for (int img = 0; img < numTestImages; img++) {
        int rows = imageArray[index + img].rows;
        int cols = imageArray[index + img].cols;
        Mat imageToShow = imageArray[index + img];
        end += rows * cols - 1;
        Mat predictionToShow = Mat(prediction, Range(start, end));
        start = end + 1;
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                if(predictionToShow.at<uchar>(i * cols + j, 0) == 1) {
                    imageToShow.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
                }
            }
        }
        imshow(title, imageToShow);
        waitKey();
    }
}

char* getCmdOption(char ** begin, char ** end, const std::string & option) {
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

int main(int argc, char * argv[]) {

    const int maxImages = 100;
    const int numTestImages = 1;
    
    Mat imageArray[maxImages], labelImageArray[maxImages];
    Mat featureArray, labelArray;
    Mat trainFeatures, trainLabels, testFeatures, testLabels, trainFeaturesForTest, trainLabelsForTest;
    Mat testPrediction, trainPrediction;
    CvDTree dtree;
 
    //command line parsing
    char* imageTemplate = getCmdOption(argv, argv + argc, "-i");
    char* labelTemplate = getCmdOption(argv, argv + argc, "-l");
    char* algorithmString = getCmdOption(argv, argv + argc, "-a");
    enum algorithm_t {NONE, DTREE, ADABOOST, SVM, BAYES, ANN, KNN};
    algorithm_t algorithm = NONE;
    if (algorithmString) {
        if (strcmp(algorithmString, "dtree") == 0)
            algorithm = DTREE;
        else if (strcmp(algorithmString, "adaboost") == 0)
            algorithm = ADABOOST;
        else if (strcmp(algorithmString, "svm") == 0)
            algorithm = SVM;
        else if (strcmp(algorithmString, "bayes") == 0)
            algorithm = BAYES;
        else if (strcmp(algorithmString, "ann") == 0)
            algorithm = ANN;
        else if (strcmp(algorithmString, "knn") == 0)
            algorithm = KNN;
    }
    if (!imageTemplate || !labelTemplate || algorithm == NONE) {
        cerr << "Usage: " << argv[0] << " -i image_name_template -l label_name_template -a algorithm" << endl;
        cerr << "image_name_template and label_name_template use c-style formatting to indicate image numbers." << endl;
        cerr << "For example 'image%03d.jpg' will open a sequence of images like image001.jpg, image002.jpg ..." << endl;
        cerr << "The label images should be binary with white regions indicating positive examples." << endl;
        cerr << "Options for the alogithm include: " << endl;
        cerr << "    'dtree' - decision tree" << endl;
        cerr << "    'adaboost' - adaptive boost with decision trees" << endl;
        cerr << "    'svm' - support vector machine" << endl;
        cerr << "    'bayes' - naive bayes" << endl;
        cerr << "    'ann' - artificial neural network" << endl;
        cerr << "    'knn' - k nearest neighbor" << endl;
        return 1;
    }
    
    //features
    int numImages = loadImages(string(imageTemplate), imageArray, maxImages);
    int totalFeatures = features(imageArray, numImages, featureArray);
    
    //lables
    int numLabels = loadImages(string(labelTemplate), labelImageArray, maxImages);
    int totalLabels = labels(labelImageArray, numLabels, labelArray);
    
    //error checking
    assert(numImages == numLabels && numImages > 0 && numLabels > 0);
    for (int i = 0; i < numImages; i++) {
        assert(imageArray[i].size == labelImageArray[i].size);
    }
    assert(numImages >= numTestImages * 2);
    cout << "Loaded " << numImages << " images with " << totalLabels << " total pixels" << endl;

    //set up data matrices
    int pixelIndex[numImages+1];
    pixelIndex[0] = 0;
    for (int i = 1; i <= numImages; i++) {
        pixelIndex[i] = pixelIndex[i-1] + imageArray[i-1].rows * imageArray[i-1].cols;
    }
    int numTrainImages = numImages - numTestImages;
    trainFeatures = featureArray(Range(0, pixelIndex[numTrainImages]), Range::all());
    trainLabels = labelArray(Range(0, pixelIndex[numTrainImages]), Range::all());
    testFeatures = featureArray(Range(pixelIndex[numTrainImages], pixelIndex[numImages]), Range::all());
    testLabels = labelArray(Range(pixelIndex[numTrainImages], pixelIndex[numImages]), Range::all());
    trainFeaturesForTest = featureArray(Range(pixelIndex[numTrainImages - numTestImages], pixelIndex[numTrainImages]), Range::all());
    trainLabelsForTest = labelArray(Range(pixelIndex[numTrainImages - numTestImages], pixelIndex[numTrainImages]), Range::all());
    
    //randomly sample the data and write to file
//    const float percentNegSamplesToKeep = 0.005;
//    const float percentPosSamplesToKeep = 0.2;
    const float percentNegSamplesToKeep = 1.1; // keep all
    const float percentPosSamplesToKeep = 1.1; // keep all
    ofstream fTrainFeatures, fTestFeatures, fTrainLabels, fTestLabels;
    fTrainFeatures.open("/Users/Clay/Dropbox/GaTech/CS7641/Assignments/Assignment1/SupervisedML/trainFeatures.csv");
    fTestFeatures.open("/Users/Clay/Dropbox/GaTech/CS7641/Assignments/Assignment1/SupervisedML/testFeatures.csv");
    fTrainLabels.open("/Users/Clay/Dropbox/GaTech/CS7641/Assignments/Assignment1/SupervisedML/trainLabels.csv");
    fTestLabels.open("/Users/Clay/Dropbox/GaTech/CS7641/Assignments/Assignment1/SupervisedML/testLabels.csv");
    Mat random(trainLabels.size(), CV_8UC1);
    randu(random, 0, 256);
    for (int i = 0; i < trainLabels.rows; i++) {
        if ((trainLabels.at<uchar>(i,0) != 1 &&
            random.at<uchar>(i, 0) < percentNegSamplesToKeep * 256) ||
            (trainLabels.at<uchar>(i,0) == 1 &&
            random.at<uchar>(i, 0) < percentPosSamplesToKeep * 256)) {
            fTrainLabels << trainLabels.at<int>(i, 0) << endl;
            for (int j = 0; j < trainFeatures.cols; j++) {
                fTrainFeatures << trainFeatures.at<float>(i,j);
                if (j != trainFeatures.cols - 1) {
                    fTrainFeatures << ",";
                }
            }
            fTrainFeatures << endl;
        }
    }
    random.resize(testLabels.rows);
    randu(random, 0, 256);
    for (int i = 0; i < testLabels.rows; i++) {
        if ((testLabels.at<uchar>(i,0) != 1 &&
             random.at<uchar>(i, 0) < percentNegSamplesToKeep * 256) ||
            (testLabels.at<uchar>(i,0) == 1 &&
             random.at<uchar>(i, 0) < percentPosSamplesToKeep * 256)) {
            fTestLabels << testLabels.at<int>(i, 0) << endl;
            for (int j = 0; j < testFeatures.cols; j++) {
                fTestFeatures << testFeatures.at<float>(i,j);
                if (j != testFeatures.cols - 1) {
                    fTestFeatures << ",";
                }
            }
            fTestFeatures << endl;
        }
    }
    fTrainFeatures.close();
    fTrainLabels.close();
    fTestFeatures.close();
    fTestLabels.close();
    
    //train and predict
    clock_t startTime = clock();
    switch (algorithm) {
        case DTREE:
            testPrediction = decisionTree(trainFeatures, trainLabels, testFeatures);
            trainPrediction = decisionTree(trainFeatures, trainLabels, trainFeaturesForTest);
            break;
        case ADABOOST:
            testPrediction = adaBoost(trainFeatures, trainLabels, testFeatures);
            trainPrediction = adaBoost(trainFeatures, trainLabels, trainFeaturesForTest);
            break;
        case SVM:
            testPrediction = svm(trainFeatures, trainLabels, testFeatures);
            trainPrediction = svm(trainFeatures, trainLabels, trainFeaturesForTest);
            break;
        case ANN:
            testPrediction = mlp(trainFeatures, trainLabels, testFeatures);
            trainPrediction = mlp(trainFeatures, trainLabels, trainFeaturesForTest);
            break;
        case KNN:
            testPrediction = knn(trainFeatures, trainLabels, testFeatures);
            trainPrediction = knn(trainFeatures, trainLabels, trainFeaturesForTest);
            break;
        case BAYES:
            testPrediction = bayes(trainFeatures, trainLabels, testFeatures);
            trainPrediction = bayes(trainFeatures, trainLabels, trainFeaturesForTest);
            break;
        default:
            cerr << "Unknown algorithm." << endl;
            return 1;
    }
    cout << "Duration for two train / predict cycles: " << (float) (clock() - startTime) / CLOCKS_PER_SEC << endl;
  
    evaluate(testPrediction, testLabels, "Testing Data");
    evaluate(trainPrediction, trainLabelsForTest, "Training Data");
  
    displayResult(imageArray, numImages - numTestImages, numTestImages, testPrediction, "Testing Data");
    displayResult(imageArray, numImages - numTestImages * 2, numTestImages, trainPrediction, "Training Data");
    
	return 0;
}