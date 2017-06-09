package utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import Jama.Matrix;

public class FileUtils {

	/**
	 * Read a matrix from a comma sperated file
	 * 
	 * @param fileName
	 * @return
	 */
	public static Matrix readMatrix(String fileName) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			List<double[]> data_array = new ArrayList<double[]>();

			String line;
			while ((line = reader.readLine()) != null) {
				String fields[] = line.split(",");
				double data[] = new double[fields.length];
				for (int i = 0; i < fields.length; ++i) {
					data[i] = Double.parseDouble(fields[i]);
				}
				data_array.add(data);
			}

			if (data_array.size() > 0) {
				int cols = data_array.get(0).length;
				int rows = data_array.size();
				Matrix matrix = new Matrix(rows, cols);
				for (int r = 0; r < rows; ++r) {
					for (int c = 0; c < cols; ++c) {
						matrix.set(r, c, data_array.get(r)[c]);
					}
				}
				return matrix;
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		return new Matrix(0, 0);
	}

	public static void readFile(String path) throws IOException {
		BufferedReader readWithBuffer = null;
		String line;
		try {
			readWithBuffer = new BufferedReader(new FileReader(path));
		} catch (FileNotFoundException exc) {
			System.out.println("Erreur lors de l'ouverture du fichier: \n" + path);
		}

		while ((line = readWithBuffer.readLine()) != null) {
			// Traitement
			System.out.println(line);
		}
		readWithBuffer.close();
	}

	public static void writeFile(String path, Matrix weights) throws IOException {
		String theta = "";
		for (int i = 0; i < weights.getRowDimension(); i++) {

			if (i == 0) {
				theta = String.valueOf(weights.get(i, 0));
			} else {
				theta += "," + String.valueOf(weights.get(i, 0));
			}

		}

		List<String> lines = Arrays.asList(theta);
		Path file = Paths.get("results\\" + path);
		Files.write(file, lines, Charset.forName("UTF-8"));
	}

	public static void main(String[] args) throws IOException {
		FileUtils.readFile("C:\\Users\\wzeghdaoui\\Personal workspace\\MachineLearningAlgorithms\\resource\\test.txt");
	}
}
