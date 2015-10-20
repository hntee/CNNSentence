package common;


import java.util.ArrayList;

import java.util.List;

import java.util.Random;



import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.IntCounter;



public class Util {
	public static void main(String[] args) throws Exception {
		int a  = 10;
		a -= -3+4;
		System.out.println(a);
		
	}
	
	public static double exp(double x) {
		if(x>50) x=50;
		else if(x<-50) x=-50;
		return Math.exp(x);
	}
	
	public static double sigmoid(double x) {
		if(x>50) x=50;
		else if(x<-50) x=-50;
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


