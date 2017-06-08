package logisticRegression;

import Jama.Matrix;
import utils.FileUtils;
import utils.MLUtils;
import utils.MatriceUtils;

public class LROneVsAll {

	public Double lambda;
	public String trainingFile;
	public String testingFile;
	public int nb_iterations;
	public Double learning_rate;

	public LROneVsAll() {
		this.lambda = 0.02;
		trainingFile = "datasets/logistic_regression_train.data";
		testingFile = "datasets/logistic_regression_test.data";
		this.nb_iterations = 100000;
		this.learning_rate = 0.1;
	}

	private double sigmoidFunction(double z) {
		return 1 / (1 + Math.exp(-z));
	}

	private double hypothesis(Matrix data, int nb_row, Matrix weights) {
		double res = 0.0;
		for (int i = 0; i < data.getColumnDimension(); i++) {
			res += data.get(nb_row, i) * weights.get(0, i);
		}
		return sigmoidFunction(res);
	}

	private double costFunction(Matrix data, Matrix targets) {

		int rows = data.getRowDimension();
		double tmp = 0.0;
		for (int i = 0; i < rows; i++) {
			tmp += costFunction_aux(data, targets, i);
		}
		return (-1 / rows) * tmp;
	}

	private double costFunction_aux(Matrix data, Matrix targets, int nb_row) {
		return -targets.get(0, nb_row) * Math.log(hypothesis(data, 0, targets))
				- ((1 - targets.get(0, nb_row)) * Math.log(1 - hypothesis(data, 0, targets)));
	}

	private double sumErrorByX(Matrix data, int rows, Matrix weights, Matrix targets, int nb_row) {

		double res = 0.0;
		for (int i = 0; i < rows; i++) {
			res += (hypothesis(data, i, weights) - targets.get(i, 0)) * data.get(i, nb_row);
		}
		return res;

	}

	private Matrix trainLinearRegressionModel(Matrix data, Matrix targets, Double lambda, double learning_rate,
			int nb_iterations) {

		int column = data.getColumnDimension();
		int rows = data.getRowDimension();
		Matrix weights = new Matrix(1, column);
		Matrix tmp_weights = new Matrix(1, column);

		for (int i = 0; i < nb_iterations; i++) {

			for (int j = 0; j < weights.getColumnDimension(); j++) {
				/** h(x(i)) - y(i)) * x(i)j */
				double sumError = sumErrorByX(data, rows, tmp_weights, targets, j);
				double tmp = weights.get(0, j) - ((learning_rate / rows) * (sumError));
				tmp_weights.set(0, j, tmp);
			}

			for (int j = 0; j < weights.getColumnDimension(); j++) {
				weights.set(0, j, tmp_weights.get(0, j));
			}
		}
		return weights.transpose();
	}



	
	/**
	 * test the model using the weights trained using logistic regression
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

		return error / (2 * row);
	}

	public static void main(String[] args) {

		LROneVsAll graD = new LROneVsAll();

		try {
			Matrix training = FileUtils.readMatrix(graD.trainingFile);
			Matrix testing = FileUtils.readMatrix(graD.testingFile);

			/** get the actual features, meanwhile add a N*1 column vector with value being all 1 as the first column of the features */
			/** Construire les matrices X pour le training set et le test set. */
			Matrix trainingData = MLUtils.getDataPoints(training);
			Matrix testingData = MLUtils.getDataPoints(testing);

			/** Construire les matrices y pour le training set et le test set. */
			Matrix trainingTargets = MLUtils.getTargets(training);
			Matrix testingTargets = MLUtils.getTargets(testing);

			/** Train the model. */
			Matrix weights = graD.trainLinearRegressionModel(trainingData, trainingTargets, graD.lambda, graD.learning_rate, graD.nb_iterations);

			/** Evaluate the model using training and testing data. */
			double training_error = graD.evaluateLinearRegressionModel(trainingData, trainingTargets, weights);
			double testing_error = graD.evaluateLinearRegressionModel(testingData, testingTargets, weights);

			 System.out.println(training_error);
			 System.out.println(testing_error);

		} catch (Exception e) {
			System.out.println("Une erreur lors du calcul du gradient descent");
		}

	}

}
