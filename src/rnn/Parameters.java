package rnn;



import java.io.Serializable;
import java.util.Arrays;
import java.util.Properties;
import edu.stanford.nlp.util.PropertiesUtils;


public class Parameters {

	/**
	  *   Out-of-vocabulary token string.
	  */
	  public static final String UNKNOWN = "-UNKN-";
	  public static final String SEPARATOR = "###################";
	
	  /**
	   * Refuse to train on words which have a corpus frequency less than
	   * this number.
	   */
	  public int wordCutOff = 1;
	
	  /**
	   * Model weights will be initialized to random values within the
	   * range {@code [-initRange, initRange]}.
	   */
	  public double initRange = 0.01;
	
	  /**
	   * Maximum number of iterations for training
	   */
	  public int maxIter = 1000;
	  public int evalPerIter = 100;
	  	
	  	  
	  /**
	   * Dimensionality of the word embeddings used
	   */
	  public int embeddingSize = 50;

	  
	  // softmax
	  public int outputSize = 2;
	  
	  public int hiddenSize = 30;
	  
	  /*
	   *  the step we look back when bptt starts, at least 1
	   *  if 1, we need s(t) , s(t-1)
	   *  if 2, we need s(t), s(t-1), s(t-2)
	   */
	  public int bpttStep = 2;
	  
	  public double alpha = 0.01;
	  public double beta = 1e-8;
	  
	  public boolean sentenceIndependent = true;
	  
	  public double s0InitValue = 0;
		
	  public Parameters(Properties properties) {
	    setProperties(properties);
	  }
	
	  private void setProperties(Properties props) {
		wordCutOff = PropertiesUtils.getInt(props, "wordCutOff", wordCutOff);
		initRange = PropertiesUtils.getDouble(props, "initRange", initRange);
		maxIter = PropertiesUtils.getInt(props, "maxIter", maxIter);
		embeddingSize = PropertiesUtils.getInt(props, "embeddingSize", embeddingSize);
		evalPerIter = PropertiesUtils.getInt(props, "evalPerIter", evalPerIter);
		outputSize = PropertiesUtils.getInt(props, "outputSize", outputSize);
		hiddenSize = PropertiesUtils.getInt(props, "hiddenSize", hiddenSize);
		bpttStep = PropertiesUtils.getInt(props, "bpttStep", bpttStep);
		alpha = PropertiesUtils.getDouble(props, "alpha", alpha);
		beta = PropertiesUtils.getDouble(props, "beta", beta);
		sentenceIndependent = PropertiesUtils.getBool(props, "sentenceIndependent", sentenceIndependent);
		s0InitValue = PropertiesUtils.getDouble(props, "s0InitValue", s0InitValue);
	  }
	
	 	
	  public void printParameters() {
		System.out.printf("wordCutOff = %d%n", wordCutOff);
		System.out.printf("initRange = %.2g%n", initRange);
		System.out.printf("maxIter = %d%n", maxIter);
		
		System.out.printf("embeddingSize = %d%n", embeddingSize);
		System.out.printf("evalPerIter = %d%n", evalPerIter);
		System.out.printf("outputSize = %d%n", outputSize);
		System.out.printf("hiddenSize = %d%n", hiddenSize);
		
		System.out.printf("bpttStep = %d%n", bpttStep);
		System.out.printf("alpha = %.2g%n", alpha);
		System.out.printf("beta = %.2g%n", beta);
		
		System.out.printf("sentenceIndependent = %b%n", sentenceIndependent);
		System.out.printf("s0InitValue = %.2g%n", s0InitValue);
		
		System.out.println(SEPARATOR);
	  }
}
