


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

	  		
	  public Parameters(Properties properties) {
	    setProperties(properties);
	  }
	
	  private void setProperties(Properties props) {
		wordCutOff = PropertiesUtils.getInt(props, "wordCutOff", wordCutOff);
		initRange = PropertiesUtils.getDouble(props, "initRange", initRange);
		maxIter = PropertiesUtils.getInt(props, "maxIter", maxIter);
		adaEps = PropertiesUtils.getDouble(props, "adaEps", adaEps);
		adaAlpha = PropertiesUtils.getDouble(props, "adaAlpha", adaAlpha);
		regParameter = PropertiesUtils.getDouble(props, "regParameter", regParameter);
		pooling = PropertiesUtils.getInt(props, "pooling", pooling);
		embeddingSize = PropertiesUtils.getInt(props, "embeddingSize", embeddingSize);
		evalPerIter = PropertiesUtils.getInt(props, "evalPerIter", evalPerIter);
		outputSize = PropertiesUtils.getInt(props, "outputSize", outputSize);
		epsilonGradientCheck = PropertiesUtils.getDouble(props, "epsilonGradientCheck", epsilonGradientCheck);
		filterNumber = PropertiesUtils.getInt(props, "filterNumber", filterNumber);
		filterMapNumber = PropertiesUtils.getInt(props, "filterMapNumber", filterMapNumber);
		batchSize = PropertiesUtils.getInt(props, "batchSize", batchSize);
		
		String temp = PropertiesUtils.getString(props, "filterWindowSize", "");
		String[] temp1 = temp.split(",");
		filterWindowSize = new int[temp1.length];
		for(int i=0;i<filterWindowSize.length;i++)
			filterWindowSize[i] = Integer.parseInt(temp1[i]);
	  }
	
	 	
	  public void printParameters() {
		System.out.printf("wordCutOff = %d%n", wordCutOff);
		System.out.printf("initRange = %.2g%n", initRange);
		System.out.printf("maxIter = %d%n", maxIter);
		System.out.printf("adaEps = %.2g%n", adaEps);
		System.out.printf("adaAlpha = %.2g%n", adaAlpha);
		System.out.printf("regParameter = %.2g%n", regParameter);
		System.out.printf("pooling = %d%n", pooling);
		System.out.printf("embeddingSize = %d%n", embeddingSize);
		System.out.printf("evalPerIter = %d%n", evalPerIter);
		System.out.printf("outputSize = %d%n", outputSize);
		System.out.printf("epsilonGradientCheck = %.2g%n", epsilonGradientCheck);
		System.out.printf("filterNumber = %d%n", filterNumber);
		System.out.printf("filterMapNumber = %d%n", filterMapNumber);
		System.out.printf("batchSize = %d%n", batchSize);
		System.out.println("filterWindowSize "+Arrays.toString(filterWindowSize));
		
		System.out.println(SEPARATOR);
	  }
}
