import java.io.Serializable;
import java.util.List;

import edu.stanford.nlp.ling.CoreLabel;
import gnu.trove.TIntArrayList;

public class Sentence implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 2769581370128602800L;
	public List<CoreLabel> tokens;
	public TIntArrayList ids; // id in the E which corresponds to each token
	public int polarity;
}
