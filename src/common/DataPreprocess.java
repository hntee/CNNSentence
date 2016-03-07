package common;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;

public class DataPreprocess {
	private static Map<String, Integer> typeMap = new HashMap<String, Integer>();

	private static final String ARTICLEPATH = "data/20news-bydate";
	private static final String TRAIN_PATH = ARTICLEPATH + "/20news-bydate-train";
	private static final String TEST_PATH = ARTICLEPATH + "/20news-bydate-test";
	private static String delimiter;
	
	public static List<Sentence> trainData = new ArrayList<>();
	public static List<Sentence> testData = new ArrayList<>();
	
	private static void initTypeMap() {
		File folder = new File(TRAIN_PATH);
		File[] listOfFiles = folder.listFiles();

	    for (int i = 0; i < listOfFiles.length; i++) {
	      String name = listOfFiles[i].getName();
	      typeMap.put(name, i);
	    }
	}
	
	private static <T> int checkType(T path) {
//		System.out.println(path);
		String categoryName =  path.toString().split(delimiter)[3];
//		System.out.println(categoryName);
		return typeMap.get(categoryName);

	}
	

	private static <E> void processData(String path, List<Sentence> trainData, List<Sentence> testData) {
		Sentence sent = new Sentence();
		sent.setCategory(checkType(path));
		PTBTokenizer<CoreLabel> ptbt;
		try {
			ptbt = new PTBTokenizer<>(new FileReader(path), new CoreLabelTokenFactory(), "");

			sent.tokens = ptbt.tokenize();

			if (path.contains("train")) {
				trainData.add(sent);
			} else if (path.contains("test")){
				testData.add(sent);
			}
			
			
//			System.out.println(trainData.size());
//			System.out.println(testData.size());
			

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}
	
	private static void initData() throws IOException {

		Stream<Path> paths = Files.walk(Paths.get(ARTICLEPATH));
		paths.filter(p -> p.toString().split(delimiter).length > 4)
             .map(p -> p.toString())
             .forEach(p -> processData(p, trainData, testData));
				               
	}
	  
	private static void initDelimiter() {
		String os = System.getProperty("os.name");
		if (os.contains("Windows")) {
			delimiter = "\\\\";
		} else {
			delimiter = "/";
		}
	}
	
	public static void init() throws IOException {
		initDelimiter();
		initTypeMap();
		initData();
	}
		
	public static void chop(int size) {
		trainData = Util.getRandomSubList(trainData, size);
		testData = Util.getRandomSubList(testData, size);
	}
}
