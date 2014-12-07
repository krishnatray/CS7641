package assignment2;


import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import shared.ConvergenceTrainer;
import shared.DataSet;
import shared.Instance;
import shared.SumOfSquaresError;
import util.linalg.DenseVector;
import util.linalg.Vector;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import assignment2.CSVfile;

public class NNWeights {
    public static void main(String[] args) {
    	
    	// load data
    	CSVfile testLabelsFile = new CSVfile("/Users/Clay/Dropbox/GaTech/CS7641/Assignments/Assignment1/SupervisedML/testLabels_small.csv");
    	CSVfile testFeaturesFile = new CSVfile("/Users/Clay/Dropbox/GaTech/CS7641/Assignments/Assignment1/SupervisedML/testFeatures_small.csv");
    	CSVfile trainLabelsFile = new CSVfile("/Users/Clay/Dropbox/GaTech/CS7641/Assignments/Assignment1/SupervisedML/trainLabels_small.csv");
    	CSVfile trainFeaturesFile = new CSVfile("/Users/Clay/Dropbox/GaTech/CS7641/Assignments/Assignment1/SupervisedML/trainFeatures_small.csv");
    	List<ArrayList<Double>> testLabels = testLabelsFile.read();
    	List<ArrayList<Double>> testFeatures = testFeaturesFile.read();
    	List<ArrayList<Double>> trainLabels = trainLabelsFile.read();
    	List<ArrayList<Double>> trainFeatures = trainFeaturesFile.read();
       	System.out.println("Loaded " + trainLabels.size() + " train samples and " + testLabels.size() + " test samples");

        // create network
       	BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
        Instance[] patterns = new Instance[trainLabels.size()];
        for (int i = 0; i < patterns.length; i++) {
            double[] trainFeaturesArray = convertDoubleListToArray(trainFeatures.get(i));
            double[] trainLabelsArray = convertDoubleListToArray(trainLabels.get(i));
        	patterns[i] = new Instance(trainFeaturesArray);
            patterns[i].setLabel(new Instance(trainLabelsArray));
        }
        BackPropagationNetwork network = factory.createClassificationNetwork(
        								 new int[] { 6, 10, 15, 1 });
                
        // train network via back propagation
        DataSet set = new DataSet(patterns);
        ConvergenceTrainer trainer = new ConvergenceTrainer(
        							 new BatchBackPropagationTrainer(
        							 set, 
        							 network,
        							 new SumOfSquaresError(), 
        							 new RPROPUpdateRule()));
        trainer.train();
        System.out.println("Convergence in " + trainer.getIterations() + " iterations");
 
        double[] weights1 = network.getWeights();
        System.out.println("numWeights: " + weights1.length);
        for (int i = 0; i < weights1.length; i++) {
        	System.out.println(weights1[i]);        	
        }
        
        // run the network over the test features
    	List<Double> results = new ArrayList<Double>();
        for (int i = 0; i < testLabels.size(); i++) {
        	network.setInputValues(convertDoubleListToVector(testFeatures.get(i)));
            network.run();
            results.add(network.getOutputValues().get(0));
//            System.out.println("label: " + testLabels.get(i) + " result: " + results.get(i));
        }
        
        evaluateResults(results, testLabels);
    }
    
    // convert an ArrayList<Double> to a double[]    
    public static double[] convertDoubleListToArray(List<Double> list) {
        double[] ret = new double[list.size()];
        Iterator<Double> iterator = list.iterator();
        for (int i = 0; i < ret.length; i++) {
            ret[i] = iterator.next().intValue();
        }
        return ret;
    }
    
    // convert an ArrayList<Double> to a util.linalg.Vector
    public static Vector convertDoubleListToVector(List<Double> list) {
		return new DenseVector(convertDoubleListToArray(list));
    }
    
    // convert util.linalg.Vector to ArrayList<Double>
    public static List<Double> convertVectorToDoubleList(Vector vector) {
    	List<Double> ret = new ArrayList<Double>();
        for (int i = 0; i < vector.size(); i++) {
            ret.add(vector.get(i));
        }
    	return ret;
    }
    
    public static void evaluateResults (List<Double> results, List<ArrayList<Double>> labels) {
    	for (double threshold = 0.4; threshold < 0.5; threshold += 0.1) {
	    	int tp = 0, tn = 0, fp = 0, fn = 0, pos = 0, neg = 0;	
    		for (int i = 0; i < results.size(); i++) {
	    		boolean p = results.get(i) < threshold ? false : true;
	    		boolean a = labels.get(i).get(0) < 0.5 ? false : true;
	    		if (p && a) {
	    			tp++;
	    		} else if (!p && !a) { 
	    			tn++;
	    		} else if (!p && a) {
	    			fn++;
	    		} else {
	    			fp++;
	    		}
	    		if (a) {
	    			pos++;
	    		} else {
	    			neg++;
	    		}
	    	}
    	    float tpr = (float) tp / pos;
    	    float fpr = (float) fp / neg;
    	    float accuracy = (float) (tp + tn) / (tp + tn + fp + fn);
    	    float precision = (float) tp / (tp + fp);
    	    float recall = (float) tp / (tp + fn);
    	    float F = (float) (2.0 * (precision * recall) / (precision + recall));
    	    System.out.println("Threshold: " + threshold);
    	    System.out.println("  True Positive Rate: " + tpr + " False Positive Rate: " + fpr);
    	    System.out.println("  Accuracy: " + accuracy + " Recall: " + recall + " Precision: " + precision + " F1: " + F);
    	}
    }
}

	