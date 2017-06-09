package utils;

import Jama.Matrix;

public class MLUtils {

	/**
	 * the last column correspond to the target Hence, remove target values from
	 * the last column of a data set.
	 * <p>
	 * Meanwhile, we need to add 1 to 0 column of each row as a bias term
	 * 
	 * @param data_set
	 * @return
	 */
	public static Matrix getDataPoints(Matrix data_set) {
		Matrix features = data_set.getMatrix(0, data_set.getRowDimension() - 1, 0, data_set.getColumnDimension() - 2);
		int rows = features.getRowDimension();
		int cols = features.getColumnDimension() + 1;
		Matrix modifiedFeatures = new Matrix(rows, cols);
		for (int r = 0; r < rows; ++r) {
			for (int c = 0; c < cols; ++c) {
				if (c == 0) {
					modifiedFeatures.set(r, c, 1.0);
				} else {
					modifiedFeatures.set(r, c, features.get(r, c - 1));
				}
			}
		}
		return modifiedFeatures;
	}

	/**
	 * Returns the target values from the last column of a data set.
	 * 
	 * @param data_set
	 * @return
	 */
	public static Matrix getTargets(Matrix data_set) {
		return data_set.getMatrix(0, data_set.getRowDimension() - 1, data_set.getColumnDimension() - 1,
				data_set.getColumnDimension() - 1);
	}

}
