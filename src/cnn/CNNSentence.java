package cnn;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Random;

import cn.fox.stanford.Tokenizer;
import common.DataPreprocess;
import common.Sentence;
import common.Tool;
import common.Util;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.PropertiesUtils;
import gnu.trove.TIntArrayList;
import gnu.trove.TObjectIntHashMap;




public class CNNSentence implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -804489238272372999L;

	public static void main(String[] args) throws Exception {
		FileInputStream fis = new FileInputStream(args[0]);
		Properties properties = new Properties();
		properties.load(fis);    
		fis.close();
		boolean debug = Boolean.parseBoolean(args[1]);

		Random random = new Random();
		String embedFile = PropertiesUtils.getString(properties, "embedFile", ""); 

		Parameters parameters = new Parameters(properties);
		parameters.printParameters();

		Tool tool = new Tool();
		tool.tokenizer = new Tokenizer(true, ' ');	

		/*
		 *  load all the positive and negative sentences.
		 *  For simpleness, we only take the first MAX_SENTENCE sentences in the positive and negative files
		 */
		DataPreprocess.init();
		DataPreprocess.chop(100);
		
		List<Sentence> trainSet = DataPreprocess.trainData;
		List<Sentence> testSet = DataPreprocess.testData;


		CNNSentence cnnSentence = new CNNSentence(parameters);
		cnnSentence.trainAndTest(trainSet, testSet, embedFile, debug);




	}

	public CNN cnn;
	public Parameters parameters;
	public List<String> knownWords;
	public TObjectIntHashMap<String> wordIDs;

	public CNNSentence(Parameters parameters) {
		this.parameters = parameters;
	}

	public void trainAndTest(List<Sentence> trainSet, List<Sentence> testSet, String embedFile, boolean debug) throws Exception {
		List<String> word = new ArrayList<>();

		for(Sentence sent:trainSet) {
			for(CoreLabel token:sent.tokens) {
				word.add(token.word()); 

			}
		}

		this.knownWords = Util.generateDict(word, this.parameters.wordCutOff);
		this.knownWords.add(0, Parameters.UNKNOWN);
		this.knownWords.add(1, Parameters.PADDING);
		this.wordIDs = new TObjectIntHashMap<String>();
		int m = 0;
		for (String w : this.knownWords)
			this.wordIDs.put(w, (m++));

		System.out.println("#Word: " + this.knownWords.size());

		// fill training with word ids
		this.fillWordID(trainSet);

		// E is the embedding matrix for every word in the given text
		double[][] E = new double[this.knownWords.size()][this.parameters.embeddingSize];
		
		Random random = new Random(System.currentTimeMillis());
		if(embedFile!=null && !embedFile.isEmpty()) { // try to load off-the-shelf embeddings
			TObjectIntHashMap<String> embedID = new TObjectIntHashMap<String>();
			BufferedReader input = null;

			input = IOUtils.readerFromString(embedFile);
			List<String> lines = new ArrayList<String>();
			for (String s; (s = input.readLine()) != null; ) {
				lines.add(s);
			}


			String[] splits = lines.get(0).split("\\s+");

			int nWords = Integer.parseInt(splits[0]);
			int dim = Integer.parseInt(splits[1]);
			lines.remove(0);
			double[][] embeddings = new double[nWords][dim];


			for (int i = 0; i < lines.size(); ++i) {
				splits = lines.get(i).split("\\s+");
				embedID.put(splits[0], i);
				for (int j = 0; j < dim; ++j)
					embeddings[i][j] = Double.parseDouble(splits[j + 1]);
			}

			// using loaded embeddings to initial E

			for (int i = 0; i < E.length; ++i) {
				int index = -1;
				if (i < this.knownWords.size()) {
					String str = this.knownWords.get(i);
					//NOTE: exact match first, and then try lower case..
					if (embedID.containsKey(str)) index = embedID.get(str);
					else if (embedID.containsKey(str.toLowerCase())) index = embedID.get(str.toLowerCase());
				}

				if (index >= 0) {
					for (int j = 0; j < E[0].length; ++j)
						E[i][j] = embeddings[index][j];
				} else {
					for (int j = 0; j < E[0].length; ++j)
						E[i][j] = random.nextDouble() * this.parameters.initRange * 2 - this.parameters.initRange;
				}
			}

		} else { // initialize E randomly
			System.out.println("No Embedding File, so initialize E randomly!");
			for(int i=0;i<E.length;i++) {
				for(int j=0;j<E[0].length;j++) {
					E[i][j] = random.nextDouble() * this.parameters.initRange * 2 - this.parameters.initRange;
				}
			}
		}

		CNN cnn = new CNN(this.parameters, this, debug, E);
		this.cnn = cnn;

		double best = 0;

		for (int iter = 0; iter < this.parameters.maxIter; ++iter) {
			System.out.print("iter: ");
			System.out.println(iter);
			List<Sentence> batch = Util.getRandomSubList(trainSet, this.parameters.batchSize);

			System.out.println("processing");
			GradientKeeper keeper = cnn.process(batch);
			
			System.out.println("updating");
			cnn.updateWeights(keeper);

			if (iter>0 && iter % this.parameters.evalPerIter == 0) {
				double temp = this.evaluate(testSet);
				if(temp>best)
					best = temp;
			}
		}


		double temp = this.evaluate(testSet);
		if(temp>best)
			best = temp;
		System.out.println("best performance :"+best);
	}

	public double evaluate(List<Sentence> testSet)
			throws Exception {
		this.fillWordID(testSet);
		int correct = 0;
		for(Sentence sentence:testSet) {

			double[] Y = this.cnn.predict(sentence);

			
			int maxIndex = 0;
			double max = 0;
			for (int i = 0; i < Y.length; i++) {
			    if (Y[i] > max) {
			        max = Y[i];
			        maxIndex = i;
			    }
			}
						
//			int predicted = 0;
//			if(Y[0]>=0.5)
//				predicted = 1;

			if(sentence.category[maxIndex] == 1)
				correct++;

		}

		double rate = correct*1.0/testSet.size();
		System.out.println("correct rate: "+rate);
		return rate;
	}

	public void fillWordID(List<Sentence> dataSet) {
		for(Sentence sentence:dataSet) {
			TIntArrayList ids = new TIntArrayList();
			for(CoreLabel token:sentence.tokens) {
				ids.add(this.getWordID(token));
			}
			sentence.ids = ids;
		}
	}

	public int getWordID(CoreLabel token) {
		return this.wordIDs.containsKey(token.word()) ? this.wordIDs.get(token.word()) : this.wordIDs.get(Parameters.UNKNOWN);
	}

	public int getWordID(String s) {

		return this.wordIDs.containsKey(s) ? this.wordIDs.get(s) : this.wordIDs.get(Parameters.UNKNOWN);
	}

}


