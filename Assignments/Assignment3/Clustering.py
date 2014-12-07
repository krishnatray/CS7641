import numpy as np
import matplotlib.pyplot as plt
import colorsys
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from sklearn.cluster import KMeans, MeanShift
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import FastICA
from sklearn.utils import shuffle
from PIL import Image
from time import time
from sklearn.decomposition.kernel_pca import KernelPCA
from sklearn.decomposition.sparse_pca import SparsePCA

# see this pull request: https://github.com/scikit-learn/scikit-learn/pull/3204
from scikit_learn.sklearn.neural_network.multilayer_perceptron import MultilayerPerceptronClassifier
# from sklearn.neural_network import MultilayerPerceptronClassifier

samples = 1000  #number of datapoints to randomly sample from the full set

# reshape a numpy array into an image with optional colormapping
def recreate_image(labels, w, h, colors = None):
    if colors != None:
        image = np.zeros((h, w, colors.shape[1]))
    else:
        image = np.zeros((h, w))
    label_idx = 0
    for i in range(h):
        for j in range(w):
            if colors !=None:
                image[i][j] = colors[labels[label_idx]]
            else:
                image[i][j] = labels[label_idx]
            label_idx += 1
    return image

# plot an image with optional colormapping
def image_show(title, img, colormap=None):
    plt.figure()
    plt.title(title)
    plt.imshow(img, colormap)

# plot a 2D scatter graph    
def scatter(title, x, y, colors, x_label='', y_label=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.scatter(x, y, c = colors)

# plot a 3D scatter graph    
def scatter3D(title, x, y, z, colors, x_label='', y_label='', z_label=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.scatter(x, y, z, c=colors)

# generate n equally space colors for colormapping    
def get_colors(num_colors):
    colors=np.zeros((num_colors, 3))
    for i in range(num_colors):
        hue = float(i) / num_colors
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        colors[i] = colorsys.hls_to_rgb(hue, lightness, saturation)
    return colors

# evaluate the performance of a binary classifier
def evaluate_Results (name, results, labels):
    threshold = 0.5
    tp = tn = fp = fn = pos = neg = 0
    for i in range(results.size):
        p = True if results[i] > threshold else False
        a = True if labels[i] > threshold else False
        if (p and a):
            tp += 1
        elif (not p and not a):
            tn += 1
        elif (not p and a):
            fn += 1
        else:
            fp += 1
        if (a):
            pos += 1
        else:
            neg += 1
    tpr = float(tp) / pos
    fpr = float(fp) / neg
    accuracy = float(tp + tn) / (tp + tn + fp + fn)
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    F = (2.0 * (precision * recall) / (precision + recall))
    print "---", name, "---"
    print "  True Positive Rate: %0.3f False Positive Rate %0.3f" % (tpr, fpr)
    print "  Accuracy: %0.3f Recall: %0.3f Precision: %0.3f F1: %0.3f" % (accuracy, recall, precision, F)

# generate an equally biased pos/neg, random subsample of the feature/label matrices        
def subsample(features, labels, n_samples):
    feature_shuffle, label_shuffle = shuffle(features, labels, random_state=0)
    i = s = 0
    label_sample = np.zeros(n_samples, dtype = np.int8)
    feature_sample = np.zeros((n_samples, features.shape[1]), dtype = np.int8)
    while i < label_shuffle.size and s < n_samples / 2:
        if label_shuffle[i] > 0.5:
            label_sample[s] = label_shuffle[i]
            feature_sample[s] = feature_shuffle[i]
            s += 1
        i += 1
    i = 0
    while i < label_shuffle.size and s < n_samples:
        if label_shuffle[i] <= 0.5:
            label_sample[s] = label_shuffle[i]
            feature_sample[s] = feature_shuffle[i]
            s += 1
        i += 1
    feature_sample, label_sample = shuffle(feature_sample, label_sample, random_state=0)
    return feature_sample, label_sample

# K-Means
def KMeans_(clusters, model_data, prediction_data = None):
    t0 = time()
    kmeans = KMeans(n_clusters=clusters).fit(model_data)
    if prediction_data == None:
        labels = kmeans.predict(model_data)
    else:
        labels = kmeans.predict(prediction_data)
    print "K Means Time: %0.3f" % (time() - t0)
    return labels

# Expectation Maximization
def EM(clusters, model_data, prediction_data = None):
    t0 = time()
    em = mixture.GMM(n_components=clusters).fit(model_data)
    if prediction_data == None:
        labels = em.predict(model_data)
    else:
        labels = em.predict(prediction_data)
    print "EM Time: %0.3f" % (time() - t0) 
    return labels

# Mean Shift
def mean_shift(model_data, prediction_data = None):
    t0 = time()
    ms = MeanShift().fit(model_data)
    if prediction_data == None:
        labels = ms.predict(model_data)
    else:
        labels = ms.predict(prediction_data)
    means = ms.cluster_centers_
    print "Number of Means:", means.shape[0] 
    print "Mean Shift Time: %0.3f" % (time() - t0)
    return labels, means    

# PCA
def PCA_(model_data, components = None, transform_data = None):
    t0 = time()
    pca = PCA(n_components=components)
    if transform_data == None:
        projection = pca.fit_transform(model_data)
    else:
        pca.fit(model_data)
        projection = pca.transform(transform_data)
    print "PCA Explained Variance: ", pca.explained_variance_ratio_
    print "PCA Time: %0.3f" % (time() - t0)
    return projection

# Sparse PCA
def SPCA(model_data, components = None, transform_data = None):
    t0 = time()
    spca = SparsePCA(n_components=components)
    if transform_data == None:
        projection = spca.fit_transform(model_data)
    else:
        spca.fit(model_data)
        projection = spca.transform(transform_data)
    print "Sparse PCA Time: %0.3f" % (time() - t0)
    return projection

# Randomized PCA
def RPCA(model_data, components = None, transform_data = None):
    t0 = time()
    rpca = RandomizedPCA(n_components=components)
    if transform_data == None:
        projection = rpca.fit_transform(model_data)
    else:
        rpca.fit(model_data)
        projection = rpca.transform(transform_data)
    print "Randomized PCA Explained Variance: ", rpca.explained_variance_ratio_
    print "Randomized PCA Time: %0.3f" % (time() - t0)
    return projection

# ICA
def ICA(model_data, components = None, transform_data = None):
    t0 = time()
    ica = FastICA(n_components=components)
    if transform_data == None:
        projection = ica.fit_transform(model_data)
    else:
        ica.fit(model_data)
        projection = ica.transform(transform_data)
    print "ICA Time: %0.3f" % (time() - t0)
    return projection

# ICA Synthetic Signals
def ICA_synthetic():
    time = np.linspace(0, 2 * np.pi, samples)
    signal1 = np.sin(2 * time)
    signal2 = np.sign(np.cos(5 * time))
    signal1 += 0.1 * np.random.normal(size=signal1.shape)
    signal2 += 0.1 * np.random.normal(size=signal2.shape)
    observation = np.c_[signal1, signal2]
    mixing = np.array([[1, .8], [1, .5]])
    observation = np.dot(observation, mixing.T)
    ica_projection = ICA(observation, components = 2)
    plt.figure()
    plt.subplot(4,1,1)
    plt.title("Signal1")
    plt.plot(signal1)
    plt.subplot(4,1,2)
    plt.title("Signal2")
    plt.plot(signal2)
    plt.subplot(4,1,3)
    plt.title("Observation")
    plt.plot(observation)
    plt.subplot(4,1,4)
    plt.title("ICA Recovered")
    plt.plot(ica_projection)

# Neural Network
def MLP(name, training_data, training_labels, test_data, test_labels):
    t0 = time()
    mlp = MultilayerPerceptronClassifier(hidden_layer_sizes=[10, 15])
    mlp.fit(training_data, training_labels)
    prediction = mlp.predict(test_data)
    evaluate_Results(name, prediction, test_labels)
    print "MLP", name, "Time: %0.3f" % (time() - t0)
    
# load image, label, feature data, and sub-samples
img = Image.open("images/1D001.jpg")
img_np = np.array(img, dtype=np.float64) / 255;
w, h = tuple(img.size)
image_array = np.reshape(img_np, (w * h, 3))
train_features = np.loadtxt("trainFeatures.csv",  delimiter=",")
train_labels = np.loadtxt("trainLabels.csv", delimiter=",", dtype = int)
test_features = np.loadtxt("testFeatures.csv",  delimiter=",")
test_labels = np.loadtxt("testLabels.csv", delimiter=",", dtype = int)
image_array_sample = shuffle(image_array, random_state=0)[:samples]
feature_sample = shuffle(train_features, random_state=0)[:samples]
train_feature_sample, train_label_sample = subsample(train_features, train_labels, samples)
test_feature_sample, test_label_sample = subsample(test_features, test_labels, samples)

clusters = 5
colormap = get_colors(clusters)

### K-Means ###
# kmeans_image = KMeans_(clusters, image_array_sample, image_array)
# kmeans_feature = KMeans_(clusters, feature_sample, train_features)
# kmeans_image_sample = KMeans_(clusters, image_array_sample)
# kmeans_feature_sample = KMeans_(clusters, train_feature_sample)
# image_show("Original Image", img)
# image_show("K-Means from RGB K="+str(clusters), recreate_image(kmeans_image, w, h, colormap))
# image_show("K-Means from Features, K="+str(clusters), recreate_image(kmeans_feature, w, h, colormap))
# scatter3D("RGB K-Means", image_array_sample[:,0], image_array_sample[:,1], image_array_sample[:,2], colormap[kmeans_image_sample[:]], "Red", "Blue", "Green")

### EM ###
# em_image = EM(clusters, image_array_sample, image_array)
# em_feature = EM(clusters, feature_sample, train_features)
# image_show("EM from RGB, K="+str(clusters), recreate_image(em_image, w, h, colormap))
# image_show("EM from Features, K="+str(clusters), recreate_image(em_feature, w, h, colormap))

### Mean Shift ###
ms_image_labels, ms_image_means = mean_shift(image_array_sample, image_array)
ms_feature_labels, ms_feature_means = mean_shift(feature_sample, train_features)
ms_colormap_image = get_colors(ms_image_means.shape[0])
ms_colormap_feature = get_colors(ms_feature_means.shape[0])
image_show("Mean Shift from RGB, Centers="+str(ms_image_means.shape[0]), recreate_image(ms_image_labels, w, h, ms_colormap_image))
image_show("Mean Shift from Features, Centers="+str(ms_feature_means.shape[0]), recreate_image(ms_feature_labels, w, h, ms_colormap_feature))

### PCA ###
pca_image_projection = PCA_(image_array, 1)
pca_feature_projection = PCA_(train_features, 1)
pca_feature_sample_projection = PCA_(train_feature_sample)
pca_test_sample_projection = PCA_(train_features, transform_data = test_feature_sample)
image_show("PCA First Component Image", recreate_image(pca_image_projection, w, h), 'gray')
image_show("PCA First Component Feature", recreate_image(pca_feature_projection, w, h), 'gray')
scatter("PCA on Features", pca_feature_sample_projection[:,0], pca_feature_sample_projection[:,1], colormap[train_label_sample[:]], "Component 1", "Component 2")

### ICA ###
# ica_image_projection = ICA(image_array, 1)
# ica_feature_projection = ICA(train_features, 1)
# image_show("ICA First Component Image", recreate_image(ica_image_projection, w, h), 'gray')
# image_show("ICA First Component Feature", recreate_image(ica_feature_projection, w, h), 'gray')

### Randomized PCA ###
# rpca_feature_projection = RPCA(train_features)

### Sparse PCA ###
# spca_feature_projection = SPCA(train_features)
# for i in range(spca_feature_projection.shape[1]):
#     image_show("PCA Sparse Component %i" % i, recreate_image(spca_feature_projection[:,i], w, h), 'gray')

### ICA Synthetic ###
# ICA_synthetic()

### K-Means on PCA Basis ###
kmeans_pca_feature = KMeans_(3, pca_feature_projection[:,:3])
kmeans_pca_feature_sample = KMeans_(6, pca_feature_sample_projection)
kmeans_pca_test_sample = KMeans_(6, pca_test_sample_projection)
image_show("K-Means from PCA-Basis Features", recreate_image(kmeans_pca_feature, w, h, colormap))

### EM on PCA Basis ###
# em_pca_feature = EM(3, pca_feature_projection[:,:3])
# image_show("EM from PCA-Basis Features", recreate_image(em_pca_feature, w, h, colormap))

### Neural Network ###
MLP("Raw Feature Sample", train_feature_sample, train_label_sample, test_feature_sample, test_label_sample)
MLP("PCA Feature Sample", pca_feature_sample_projection[:,:1], train_label_sample, pca_test_sample_projection[:,:1], test_label_sample)
kmeans_pca_features = np.zeros((samples, 2))
kmeans_pca_features[:, 0] = pca_feature_sample_projection[:,0]
kmeans_pca_features[:, 1] = kmeans_pca_feature_sample
kmeans_pca_test = np.zeros((samples, 2))
kmeans_pca_test[:, 0] = pca_test_sample_projection[:,0]
kmeans_pca_test[:, 1] = kmeans_pca_test_sample
MLP("K-Means on PCA Feature Sample", kmeans_pca_features, train_label_sample, kmeans_pca_test, test_label_sample)

plt.show()

