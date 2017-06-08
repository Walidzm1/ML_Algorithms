package logisticRegression;

import Jama.Matrix;
import utils.FileUtils;
import utils.MLUtils;

public class LRGradientDescent {

	public Double lambda;
	public String trainingFile;
	public String testingFile;
	public int nb_iterations;
	public int nb_classes;
	public Double learning_rate;

	public LRGradientDescent() {
		this.lambda = 0.02;
		trainingFile = "datasets/logistic_regression_train.data";
		testingFile = "datasets/logistic_regression_test.data";
		this.nb_iterations = 100000;
		this.learning_rate = 0.0001;
	}

	private double sigmoidFunction(double z) {
		return 1.0 / (1.0 + Math.exp(-z));
	}

	private double hypothesis(Matrix data, int nb_row, Matrix weights) {
		double res = 0.0;
		for (int j = 0; j < data.getColumnDimension(); j++) {
			res += data.get(nb_row, j) * weights.get(j, 0);
		}
		return sigmoidFunction(res);
	}

	private double costFunction(Matrix targets, Matrix predictTargets) {

		double tmp = 0.0;
		for (int i = 0; i < targets.getRowDimension(); i++) {
			tmp += costFunction_aux(targets, predictTargets, i);
		}
		return -1 * tmp / targets.getRowDimension();
	}

	private double costFunction_aux(Matrix targets, Matrix predictTargets, int nb_row) {
		return (-1 * targets.get(nb_row, 0) * Math.log(predictTargets.get(nb_row, 0)))
				+ ((1 - targets.get(nb_row, 0)) * Math.log(1 - predictTargets.get(nb_row, 0)));
	}

	private double sumErrorByX(Matrix data, Matrix weights, Matrix targets, int nb_row) {

		int rows = data.getRowDimension();
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
		Matrix weights = new Matrix(column, 1);
		Matrix tmp_weights = new Matrix(column, 1);

		for (int i = 0; i < nb_iterations; i++) {
			for (int j = 0; j < weights.getRowDimension(); j++) {
				/** h(x(i)) - y(i)) * x(i)j */
				double sumError = sumErrorByX(data, weights, targets, j);
				double tmp = weights.get(j, 0) - ((learning_rate / rows) * (sumError));
				tmp_weights.set(j, 0, tmp);
			}
			for (int j = 0; j < weights.getRowDimension(); j++) {
				weights.set(j, 0, tmp_weights.get(j, 0));
			}
		}
		return weights;
	}

	private double evaluateLinearRegressionModel(Matrix data, Matrix targets, Matrix weights) {

		int row = data.getRowDimension();
		int column = data.getColumnDimension();
		assert row == targets.getRowDimension();
		assert column == weights.getColumnDimension();

		Matrix predictTargets = predict(data, weights);

		return costFunction(targets, predictTargets);
	}

	private Matrix predict(Matrix data, Matrix weights) {

		int row = data.getRowDimension();
		Matrix predictTargets = new Matrix(row, 1);
		for (int i = 0; i < row; i++) {
			double value = hypothesis(data, i, weights);
			predictTargets.set(i, 0, value);
		}
		return predictTargets;
	}

	public static void main(String[] args) {

		LRGradientDescent lr = new LRGradientDescent();

		try {
			Matrix training = FileUtils.readMatrix(lr.trainingFile);
			Matrix testing = FileUtils.readMatrix(lr.testingFile);

			/**
			 * get the actual features, meanwhile add a N*1 column vector with
			 * value being all 1 as the first column of the features
			 */
			/**
			 * Construire les matrices X pour le training set et le test set.
			 */
			Matrix trainingData = MLUtils.getDataPoints(training);
			Matrix testingData = MLUtils.getDataPoints(testing);

			/**
			 * Construire les matrices y pour le training set et le test set.
			 */
			Matrix trainingTargets = MLUtils.getTargets(training);
			Matrix testingTargets = MLUtils.getTargets(testing);

			/** Train the model. */
			Matrix weights = lr.trainLinearRegressionModel(trainingData, trainingTargets, lr.lambda, lr.learning_rate,
					lr.nb_iterations);
			for (int i = 0; i < weights.getRowDimension(); i++) {
				System.out.print(weights.get(i, 0) + " ");
			}
			System.out.println();

			/** Evaluate the model using training and testing data. */
			double training_error = lr.evaluateLinearRegressionModel(trainingData, trainingTargets, weights);
			double testing_error = lr.evaluateLinearRegressionModel(testingData, testingTargets, weights);

			System.out.println(training_error);
			System.out.println(testing_error);

		} catch (Exception e) {
			System.out.println("Une erreur lors du calcul du gradient descent");
		}

	}

}
