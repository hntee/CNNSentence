

import static java.util.stream.Collectors.toSet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;


import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.util.PropertiesUtils;
import gnu.trove.TIntArrayList;


public class Util {
	public static void main(String[] args) throws Exception {
		FileInputStream fis = new FileInputStream(args[0]);
		Properties properties = new Properties();
		properties.load(fis);    
		fis.close();
		
		Parameters parameters = new Parameters(properties);
		
		//prepareWord2VecCorpus(properties, parameters);
		Random random = new Random(System.currentTimeMillis());
		
		
                
		
	}
	
	public static double relu(double x) {
		return Math.max(0, x);
	}
	
	public static double sigmoid(double x) {
		return 1.0/(1+Math.exp(-x));
	}

	public static List<String> generateDict(List<String> str, int cutOff)
	  {
	    Counter<String> freq = new IntCounter<>();
	    for (String aStr : str)
	      freq.incrementCount(aStr);

	    List<String> keys = Counters.toSortedList(freq, false);
	    List<String> dict = new ArrayList<>();
	    for (String word : keys) {
	      if (freq.getCount(word) >= cutOff)
	        dict.add(word);
	    }
	    return dict;
	  }
	
	  public static <T> List<T> getRandomSubList(List<T> input, int subsetSize)
	  {
	    int inputSize = input.size();
	    if (subsetSize > inputSize)
	      subsetSize = inputSize;

	    Random random = new Random(System.currentTimeMillis());
	  
	    for (int i = 0; i < subsetSize; i++)
	    {
	      int indexToSwap = i + random.nextInt(inputSize - i);
	      T temp = input.get(i);
	      input.set(i, input.get(indexToSwap));
	      input.set(indexToSwap, temp);
	    }
	    return input.subList(0, subsetSize);
	  }
	
	
}


