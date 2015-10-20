package rnn;

public class History {
	double[] neu1;
	double[] err_neu1;
	double[] neu0;
	int wordID;
	
	public History(double[] neu1, double[] err_neu1, double[] neu0, int wordID) {
		this.neu1 = neu1;
		this.err_neu1 = err_neu1;
		this.neu0 = neu0;
		this.wordID = wordID;
	}
}
