package linearRegression;

import Jama.Matrix;
import utils.FileUtils;
import utils.MLUtils;
import utils.MatriceUtils;

import java.io.FileReader;
import java.io.BufferedReader;
import java.util.List;
import java.util.ArrayList;

/**
 * A class to train and evaluate Linear Regression models with L2-Regularization
 * @author walidzm1
 */
public class LRNormalEquation {
	public Double lambda;
	public String trainingFile;
	public String testingFile;
	
	public LRNormalEquation() {
		lambda = 0.2;
		trainingFile = "datasets/linear_regression_train.data";
		testingFile = "datasets/linear_regression_test.data";
	}
	
	/**
	 * train the model using linear regression with L2 regularizer
	 * The first column of the features should be set as all 1. because the first feature should perform as bias
	 * So we should not regularize the theta0
	 * 
	 * Linear regression with L2 regularizer has the close form as follows:
	 * 		
	 * 			w = (X^{T} * X + \lambda * I)^{-1} * X^{T} * t 
	 * 
	 * 		Linear regression using normal equation:
	 * 
	 * 		+ No need to choose alpha (learning rate).
	 * 		+ Don't need to iterate.
	 * 		- Need to compute (X^{T} * X)^{-1} (O(n^3)).
	 * 		- Slow if the number of features is very large.
	 * 
	 * @param data n * m
	 * @param targets n * 1
	 * @param lambda
	 * @return weight m * 1
	 */
	private Matrix trainLinearRegressionModel(Matrix data, Matrix targets, Double lambda) {
		int column = data.getColumnDimension();
		Matrix identity = Matrix.identity(column , column);
		identity.set(0, 0, 0);
		identity = identity.times(lambda);
		
		Matrix dataCopy = data.copy();
		Matrix transponseData = dataCopy.transpose();
		Matrix norm = transponseData.times(data);
		Matrix circular = norm.plus(identity);
		Matrix circularInverse = circular.inverse();
		Matrix former = circularInverse.times(data.transpose());
		Matrix weight = former.times(targets);
		
	    return weight;
	}
	
	/**
	 * test the model using the weights trained using linear regression with L2 regularizer
	 * 
	 * @param data n * m matrix
	 * @param targets n * 1 matrix
	 * @param weights m * 1 matrix
	 * @return
	 */
	private double evaluateLinearRegressionModel(Matrix data, Matrix targets, Matrix weights) {
		double error = 0.0;
		int row = data.getRowDimension();
		int column = data.getColumnDimension();
		assert row == targets.getRowDimension();
		assert column == weights.getColumnDimension();
		
		Matrix predictTargets = MatriceUtils.predict(data, weights);
		for (int i = 0; i < row; i++) {
			error += (predictTargets.get(i, 0) - targets.get(i, 0)) * (predictTargets.get(i, 0) - targets.get(i, 0));
		}

		/** return 0.5 * error; */
		return error / (2 * row);
	}
	
	/**
	 * args consists of the training file path and testing file path
	 * 
	 * If you want to set the lamdab (penalization coefficient), you can go to the Linear Regression constructor
	 * to update the value of lambda
	 * 
	 * In addition, you can update the training path and testing path in the same way.
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		LRNormalEquation norEq = new LRNormalEquation();
		try {
			Matrix training = FileUtils.readMatrix(norEq.trainingFile);
			Matrix testing = FileUtils.readMatrix(norEq.testingFile);
			
			/** get the actual features, meanwhile add a N*1 column vector with value being all 1 as the first column of the features */
			/** Construire les matrices X pour le training set et le test set. */
			Matrix trainingData = MLUtils.getDataPoints(training);
			Matrix testingData = MLUtils.getDataPoints(testing);
			
			/** Construire les matrices y pour le training set et le test set. */
			Matrix trainingTargets = MLUtils.getTargets(training);
		    Matrix testingTargets = MLUtils.getTargets(testing);
		    
		    /** Train the model. */
		    Matrix weights = norEq.trainLinearRegressionModel(trainingData, trainingTargets, norEq.lambda);
		    for (int i = 0; i < weights.getRowDimension(); i++) {
		    	System.out.print(weights.get(i, 0) + " " );
		    }
		    System.out.println();
		    
		    /** Evaluate the model using training and testing data. */
		    double training_error = norEq.evaluateLinearRegressionModel(trainingData, trainingTargets, weights);
		    double testing_error = norEq.evaluateLinearRegressionModel(testingData, testingTargets, weights);

		    System.out.println(training_error);
		    System.out.println(testing_error);
		} catch (Exception e) {
			System.out.println("Une erreur lors du calcul du l'equation normale");
		}
	}
	
}