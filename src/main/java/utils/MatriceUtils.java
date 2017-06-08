package utils;

import Jama.Matrix;

public class MatriceUtils {

	
	/**
	 * calcualte the predict targests given features and the learned weights
	 * 
	 * @param data a matrix with n * m 
	 * @param weights a matrix with 1 * m
	 * @return predict targets according to the weight
	 */
	public static Matrix predict(Matrix data, Matrix weights) {
		int row = data.getRowDimension();
		Matrix predictTargets = new Matrix(row, 1);
		for (int i = 0; i < row; i++) {
			double value = multiply(data.getMatrix(i, i, 0, data.getColumnDimension() -1 ), weights);
			predictTargets.set(i, 0, value);
		}
		return predictTargets;
	}
	
	/**
	 * multiply two matrix with just 1 row and seveal columns
	 * 
	 * @param data a matrix with 1 * column 
	 * @param weights a matrix with 1 * column
	 * @return
	 */
	public static Double multiply(Matrix data, Matrix weights) {
		Double sum = 0.0;
		int column = data.getColumnDimension();
		for (int i = 0; i <column; i++) {
			sum += data.get(0, i) * weights.get(i, 0);
		}
		return sum;
	}
	
}
