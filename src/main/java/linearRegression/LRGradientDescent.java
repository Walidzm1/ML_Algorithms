package linearRegression;

import Jama.Matrix;
import utils.FileUtils;
import utils.MLUtils;
import utils.MatrixUtils;

public class LRGradientDescent {

	public Double lambda;
	public String trainingFile;
	public String testingFile;
	public int nb_iterations;
	public Double learning_rate;

	public LRGradientDescent() {
		this.lambda = 0.2;
//		trainingFile = "datasets/linear_regression_train.data";
//		testingFile = "datasets/linear_regression_test.data";
		trainingFile = "datasets/linearR_train.data";
		testingFile = "datasets/linearR_test.data";
		this.nb_iterations = 1000000;
		this.learning_rate = 0.01;
	}

	private double hypothesis(Matrix data, int nb_row, Matrix weights) {
		double res = 0.0;
		for (int i = 0; i < data.getColumnDimension(); i++) {
			res += data.get(nb_row, i) * weights.get(i, 0);
		}
		return res;
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
				double right_value = 0.0;
				/** do not regularize the theta 0 */
				if (j == 0) {
					right_value = weights.get(j, 0) - ((learning_rate / rows) * sumError);
				} else {
					right_value = weights.get(j, 0) * (1 - (learning_rate * lambda / rows))
							- ((learning_rate / rows) * sumError);
				}
				tmp_weights.set(j, 0, right_value);
			}

			for (int j = 0; j < weights.getRowDimension(); j++) {
				weights.set(j, 0, tmp_weights.get(j, 0));
			}
		}
		return weights;
	}

	/**
	 * test the model using the weights trained using linear regression with L2
	 * regularizer
	 */
	private double evaluateLinearRegressionModel(Matrix data, Matrix targets, Matrix weights) {
		double error = 0.0;
		int row = data.getRowDimension();
		int column = data.getColumnDimension();
		assert row == targets.getRowDimension();
		assert column == weights.getColumnDimension();

		Matrix predictTargets = MatrixUtils.predict(data, weights);
		for (int i = 0; i < row; i++) {
			error += (predictTargets.get(i, 0) - targets.get(i, 0)) * (predictTargets.get(i, 0) - targets.get(i, 0));
		}
		return error / (2 * row);
	}

	public static void main(String[] args) {

		LRGradientDescent graD = new LRGradientDescent();
		try {
			Matrix training = FileUtils.readMatrix(graD.trainingFile);
			Matrix testing = FileUtils.readMatrix(graD.testingFile);

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
			Matrix weights = graD.trainLinearRegressionModel(trainingData, trainingTargets, graD.lambda,
					graD.learning_rate, graD.nb_iterations);
			
			FileUtils.writeFile("linear_regressoin_gradient_descent_thetas.data", weights);

			/** Evaluate the model using training and testing data. */
			double training_error = graD.evaluateLinearRegressionModel(trainingData, trainingTargets, weights);
			double testing_error = graD.evaluateLinearRegressionModel(testingData, testingTargets, weights);

			System.out.println("Training error: "+training_error);
			System.out.println("Test error: "+testing_error);
		} catch (Exception e) {
			System.out.println("Gradient descent (Linear Regression)");
		}

	}

}
