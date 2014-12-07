package assignment2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class CSVfile {
	private String fileName;
	
	public CSVfile(String fileName) {
		this.fileName = fileName;
	}

	public ArrayList<ArrayList<Double>> read() {
		BufferedReader br = null;
		ArrayList<ArrayList<Double>> data = new ArrayList<ArrayList<Double>>();
		
		try {
			br = new BufferedReader(new FileReader(fileName));
			String line;
			while ((line = br.readLine()) != null) {
				data.add(new ArrayList<Double>());
				String[] rowString = line.split(",");
				for (int i = 0; i < rowString.length; i++) {
					try {
						data.get(data.size() - 1).add(Double.parseDouble(rowString[i]));
					} catch (NumberFormatException e) {
						e.printStackTrace();						
					}
				}
			} 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}	
		}
		return data;
	}
}
