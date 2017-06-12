package logisticRegression;

import Jama.Matrix;
import utils.FileUtils;
import utils.MLUtils;

public class LRGradientDescent {

	public Double lambda;
	public String trainingFile;
	public String testingFile;
	public int nb_iterations;
	public Double learning_rate;

	public LRGradientDescent() {
		this.lambda = 0.02;
		// trainingFile = "datasets/logistic_regression_train.data";
		// testingFile = "datasets/logistic_regression_test.data";
		trainingFile = "datasets/logisticR_train.data";
		testingFile = "datasets/logisticR_test.data";
		this.nb_iterations = 10000000;
		this.learning_rate = 0.001;
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

	private double costFunction(Matrix data, Matrix targets, Matrix predictTargets, Matrix weights) {

		double tmp = 0.0;
		for (int i = 0; i < targets.getRowDimension(); i++) {
			tmp += costFunction_aux(targets, predictTargets, i);
		}
		return (-1 * tmp / targets.getRowDimension()) + regularize(data, weights);
	}

	private double costFunction_aux(Matrix targets, Matrix predictTargets, int nb_row) {

		return (targets.get(nb_row, 0) * Math.log(predictTargets.get(nb_row, 0)))
				+ ((1 - targets.get(nb_row, 0)) * Math.log(1 - predictTargets.get(nb_row, 0)));
	}

	private double regularize(Matrix data, Matrix weights) {

		double res = 0.0;
		for (int i = 0; i < weights.getRowDimension(); i++) {
			res += weights.get(i, 0) * weights.get(i, 0);
		}

		return (lambda / 2 * data.getRowDimension()) * res;
	}

	private double sumErrorByX(Matrix data, Matrix weights, Matrix targets, int nb_row) {

		int rows = data.getRowDimension();
		double res = 0.0;
		for (int i = 0; i < rows; i++) {
			res += (hypothesis(data, i, weights) - targets.get(i, 0)) * data.get(i, nb_row);
		}
		return res;

	}

	private Matrix trainLogisticRegressionModel(Matrix data, Matrix targets, Double lambda, double learning_rate,
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

	private void classify(Matrix Xs, Matrix thetas) {

		thetas = thetas.transpose();
		String _class = "";
		for (int i = 0; i < Xs.getRowDimension(); i++) {
			double tmp = -1.1;
			for (int j = 0; j < thetas.getRowDimension(); j++) {
				Matrix theta_tmp = thetas.getMatrix(j, j, 0, thetas.getColumnDimension() - 1);
				if (hypothesis(Xs, i, theta_tmp.transpose()) > tmp) {
					tmp = hypothesis(Xs, i, theta_tmp.transpose());
					_class = String.valueOf(j);
				}
			}
			int _class_opt = Integer.parseInt(_class);
			Matrix theta_opt = thetas.getMatrix(_class_opt, _class_opt, 0, thetas.getColumnDimension() - 1);
			System.out.println("class of i_" + i + " (" + Xs.get(i, 1) + "," + Xs.get(i, 2) + ") >> "
					+ classify_aux(Xs, i, theta_opt.transpose()));
		}
	}

	private double classify_aux(Matrix Xs, int nb_row, Matrix theta) {
		double res = 0.0;
		for (int j = 1; j < Xs.getColumnDimension(); j++) {
			res += Xs.get(nb_row, j) * theta.get(j, 0);
		}
		if (res >= (-1 * theta.get(0, 0))) {
			return 1;
		} else
			return 0;
	}
	
	private double evaluateLogisticRegressionModel(Matrix data, Matrix targets, Matrix weights) {

		int row = data.getRowDimension();
		int column = data.getColumnDimension();
		assert row == targets.getRowDimension();
		assert column == weights.getColumnDimension();

		Matrix predictTargets = predict(data, weights);

		return costFunction(data, targets, predictTargets, weights);
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
			Matrix weights = lr.trainLogisticRegressionModel(trainingData, trainingTargets, lr.lambda, lr.learning_rate,
					lr.nb_iterations);

			FileUtils.writeFile("logistic_regressoin_gradient_descent_thetas.data", weights);

			/** Evaluate the model using training and testing data. */
			double training_error = lr.evaluateLogisticRegressionModel(trainingData, trainingTargets, weights);
			double testing_error = lr.evaluateLogisticRegressionModel(testingData, testingTargets, weights);

			System.out.println("Training error: " + training_error);
			System.out.println("Test error: " + testing_error);

			lr.classify(testingData, weights);
			
		} catch (Exception e) {
			System.out.println("Gradient descent error (logistic regression)");
		}

	}

}
