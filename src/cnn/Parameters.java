package cnn;



import java.io.Serializable;
import java.util.Arrays;
import java.util.Properties;

import edu.stanford.nlp.util.PropertiesUtils;


public class Parameters implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = -8092716117880301475L;
	/**
	  *   Out-of-vocabulary token string.
	  */
	  public static final String UNKNOWN = "-UNKN-";
	  // sometimes the sentence is shorter than the window, so we need the padding word. 
	  public static final String PADDING = "-PAD-"; 
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
	   * An epsilon value added to the denominator of the AdaGrad
	   * expression for numerical stability
	   */
	  public double adaEps = 1e-6;
	
	  /**
	   * Initial global learning rate for AdaGrad training
	   */
	  public double adaAlpha = 0.01;
	
	  /**
	   * Regularization parameter. All weight updates are scaled by this
	   * single parameter.
	   */
	  public double regParameter = 1e-8;
	
	  // 1-avg, 2-max, 3-min
	  public int pooling = 1;
	
	  
	
	  /**
	   * Dimensionality of the word embeddings used
	   */
	  public int embeddingSize = 50;
	
	  	
	  public int filterNumber = 1;
	  
	  // the window size of each filter, so its size corresponds to 'filterNumber'
	  public int[] filterWindowSize = new int[]{3};
	  // the output node number of the filter
	  public int filterMapNumber =  300;
	  
	  // logistic regression
	  public int outputSize = 1;
	  
	  	  
	  public int batchSize = 50;
	  
	  // for gradient checking
	  public double epsilonGradientCheck = 1e-4;

	  public String articlePath;
	  public Parameters(Properties properties) {
	    this.setProperties(properties);
	  }
	
	  private void setProperties(Properties props) {
		this.wordCutOff = PropertiesUtils.getInt(props, "wordCutOff", this.wordCutOff);
		this.initRange = PropertiesUtils.getDouble(props, "initRange", this.initRange);
		this.maxIter = PropertiesUtils.getInt(props, "maxIter", this.maxIter);
		this.adaEps = PropertiesUtils.getDouble(props, "adaEps", this.adaEps);
		this.adaAlpha = PropertiesUtils.getDouble(props, "adaAlpha", this.adaAlpha);
		this.regParameter = PropertiesUtils.getDouble(props, "regParameter", this.regParameter);
		this.pooling = PropertiesUtils.getInt(props, "pooling", this.pooling);
		this.embeddingSize = PropertiesUtils.getInt(props, "embeddingSize", this.embeddingSize);
		this.evalPerIter = PropertiesUtils.getInt(props, "evalPerIter", this.evalPerIter);
		this.outputSize = PropertiesUtils.getInt(props, "outputSize", this.outputSize);
		this.epsilonGradientCheck = PropertiesUtils.getDouble(props, "epsilonGradientCheck", this.epsilonGradientCheck);
		this.filterNumber = PropertiesUtils.getInt(props, "filterNumber", this.filterNumber);
		this.filterMapNumber = PropertiesUtils.getInt(props, "filterMapNumber", this.filterMapNumber);
		this.batchSize = PropertiesUtils.getInt(props, "batchSize", this.batchSize);
		
		String temp = PropertiesUtils.getString(props, "filterWindowSize", "");
		String[] temp1 = temp.split(",");
		this.filterWindowSize = new int[temp1.length];
		for(int i=0;i<this.filterWindowSize.length;i++)
			this.filterWindowSize[i] = Integer.parseInt(temp1[i]);
	  }
	
	 	
	  public void printParameters() {
		System.out.printf("wordCutOff = %d%n", this.wordCutOff);
		System.out.printf("initRange = %.2g%n", this.initRange);
		System.out.printf("maxIter = %d%n", this.maxIter);
		System.out.printf("adaEps = %.2g%n", this.adaEps);
		System.out.printf("adaAlpha = %.2g%n", this.adaAlpha);
		System.out.printf("regParameter = %.2g%n", this.regParameter);
		System.out.printf("pooling = %d%n", this.pooling);
		System.out.printf("embeddingSize = %d%n", this.embeddingSize);
		System.out.printf("evalPerIter = %d%n", this.evalPerIter);
		System.out.printf("outputSize = %d%n", this.outputSize);
		System.out.printf("epsilonGradientCheck = %.2g%n", this.epsilonGradientCheck);
		System.out.printf("filterNumber = %d%n", this.filterNumber);
		System.out.printf("filterMapNumber = %d%n", this.filterMapNumber);
		System.out.printf("batchSize = %d%n", this.batchSize);
		System.out.println("filterWindowSize "+Arrays.toString(this.filterWindowSize));
		
		System.out.println(SEPARATOR);
	  }
}
