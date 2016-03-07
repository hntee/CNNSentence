package common;
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
	public int[] category;
		
	public int getCategory() {
		int cat = -1;
		for (int i = 0; i < this.category.length; i++) {
			if (this.category[i] == 1) 
				cat = i;
		}
		return cat;
	}

	public void setCategory(int cat) {
		this.category[cat] = 1;
	}

	public Sentence() {
		this.category = new int[20];
	}
	
	public Sentence(int dimension) {
		this.category = new int[dimension];
	}
	
	
	
	
}
