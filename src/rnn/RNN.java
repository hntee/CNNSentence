package rnn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import common.Sentence;
import common.Util;
import edu.stanford.nlp.ling.CoreLabel;

class Neu {
	public double ac; // value of this neuron
	public double er; // error of this neuron
}

public class RNN {

	public double[][] syn0; // U and W
	public double[][] syn1; // V
	
	// getLast is the newest and its size is bpttStep+1
	public LinkedList<History> history; 
	
	public Parameters parameters;
	public boolean debug;
	public double[][] E;
	
	public RNN(Parameters parameters, boolean debug, double[][] E) {
		this.parameters = parameters;
		this.debug = debug;
		this.E = E;
		
		Random random = new Random(System.currentTimeMillis());
		
			
		syn0 = new double[parameters.hiddenSize][parameters.embeddingSize+parameters.hiddenSize];
		for(int i=0;i<syn0.length;i++) {
			for(int j=0;j<syn0[0].length;j++) {
				syn0[i][j] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
			}
		}
		syn1 = new double[parameters.outputSize][parameters.hiddenSize];
		for(int i=0;i<syn1.length;i++) {
			for(int j=0;j<syn1[0].length;j++) {
				syn1[i][j] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
			}
		}
		
		history = new LinkedList<>();
	}
	
	private void initHistory() {
		history.clear();
		
		History first = new History(new double[parameters.hiddenSize], new double[parameters.hiddenSize], 
				new double[parameters.embeddingSize+parameters.hiddenSize], 0);
		history.addLast(first);
		
		for(int i=0;i<first.neu1.length;i++) {
			first.neu1[i] = 1;
		}
	}
	
	private void addHistory(double[] neu1, double[] err_neu1, double[] neu0, int wordID) {
		History current = new History(neu1, err_neu1, neu0, wordID);
		history.addLast(current);
		if(history.size() > parameters.bpttStep+1)
			history.removeFirst();
	}
	
	private double deltaTrunc(double delta) {
		// truncate the gradient of delta
		
		if(delta>15) {
			System.out.println("gradient truncated "+delta);
			return 15;
		}
		else if(delta<-15) {
			System.out.println("gradient truncated "+delta);
			return -15;
		} else
			return delta;
		
	}
	
	public double[] predict(Sentence sent) throws Exception {
		double[] neu1 = new double[parameters.hiddenSize];
		for(int tokenIdx=0; tokenIdx<sent.tokens.size();tokenIdx++) {
			double[] neu0 = new double[parameters.embeddingSize+parameters.hiddenSize];
			
			for(int i=0;i<parameters.embeddingSize;i++) {
				neu0[i] = E[sent.ids.get(tokenIdx)][i];
			}
			
			if(tokenIdx==0) {
				for(int i=0;i<parameters.hiddenSize;i++) {
					neu0[i+parameters.embeddingSize] = 1;
				}
			} else {
				for(int i=0;i<parameters.hiddenSize;i++) {
					neu0[i+parameters.embeddingSize] = neu1[i];
				}
			}
			
			
			for(int j=0;j<neu1.length;j++) {
				neu1[j] = 0;
			}
			// clear neu1 because the loop below use +=
			
			// neu0->neu1
			for(int j=0;j<neu1.length;j++) {
				for(int i=0;i<neu0.length;i++) {
					neu1[j] += syn0[j][i]*neu0[i];
				}
			}
			for(int j=0;j<neu1.length;j++) {
				neu1[j] = Util.sigmoid(neu1[j]);
			}
			
			
		}
		
		// neu1->neu2
		double[] neu2 = new double[parameters.outputSize];
		for(int k=0;k<neu2.length;k++) {
			for(int j=0;j<neu1.length;j++) {
				neu2[k] += syn1[k][j]*neu1[j];
			}
		}
		return neu2;
	}
	
	// SGD with BPTT
	public void process(List<Sentence> trainSet) throws Exception {
		
		initHistory();
		
		double loss = 0;
		
		for(Sentence sent:trainSet) {
							
			// make a gold output
			double[] gold = new double[parameters.outputSize];
			if(sent.polarity==1)
				gold[0] = 1;
			else
				gold[1] = 1;
			
			for(int tokenIdx=0; tokenIdx<sent.tokens.size();tokenIdx++) {
				/** forward **/
				// build neu0
				double[] neu0 = new double[parameters.embeddingSize+parameters.hiddenSize];
				
				for(int i=0;i<parameters.embeddingSize;i++) {
					neu0[i] = E[sent.ids.get(tokenIdx)][i];
				}
				
				double[] last = history.getLast().neu1;
				for(int i=0;i<last.length;i++) {
					neu0[i+parameters.embeddingSize] = last[i];
				}
				
				// neu0->neu1
				double[] neu1 = new double[parameters.hiddenSize];
				for(int j=0;j<neu1.length;j++) {
					for(int i=0;i<neu0.length;i++) {
						neu1[j] += syn0[j][i]*neu0[i];
					}
				}
				for(int j=0;j<neu1.length;j++) {
					neu1[j] = Util.sigmoid(neu1[j]);
				}
				// neu1->neu2
				double[] neu2 = new double[parameters.outputSize];
				for(int k=0;k<neu2.length;k++) {
					for(int j=0;j<neu1.length;j++) {
						neu2[k] += syn1[k][j]*neu1[j];
					}
				}
				double sum = 0;
				double sum1 = 0;
				for(int k=0;k<neu2.length;k++) {
					neu2[k] = Util.exp(neu2[k]);
					sum += neu2[k];
					if(gold[k]==1)
						sum1 += neu2[k];
				}
				for(int k=0;k<neu2.length;k++) {
					neu2[k] = neu2[k]/sum;
				}
				
				loss += Math.log(sum) - Math.log(sum1);
				
				/** backward **/
				// neu2 -> neu1
				double[] delta_neu2 = new double[neu2.length];
				double[] err_neu1 = new double[neu1.length];
				for(int k=0;k<neu2.length;k++) {
					delta_neu2[k] = deltaTrunc(neu2[k]-gold[k]);
					for(int j=0;j<neu1.length;j++) {
						err_neu1[j] +=  delta_neu2[k]*syn1[k][j];
						syn1[k][j] -= parameters.alpha*delta_neu2[k]*neu1[j]+parameters.beta*syn1[k][j]; // update syn1
					}
				}
				// save history
				addHistory(neu1, err_neu1, neu0, sent.ids.get(tokenIdx));
				
				// neu1 -> neu0, 
				if(history.size()==parameters.bpttStep+1) {
					// if the history of words we have processed is enough, we do a BPTT
					// but note that if bpttStep=1, we also perform the code here
					double[][] temp_syn0 = new double[syn0.length][syn0[0].length];
					double[] temp_err_neu1 = new double[err_neu1.length];
					for(int t = parameters.bpttStep; t>=1; t--) {
						History current = history.get(t);
						for(int j=0;j<temp_err_neu1.length;j++) {
							temp_err_neu1[j] += current.err_neu1[j]; // add the error of last step to the current err_neu1
						}
				
						double[] delta_neu1 = new double[current.neu1.length];
						double[] err_neu0 = new double[current.neu0.length];
						for(int j=0;j<current.neu1.length;j++) {
							delta_neu1[j] = deltaTrunc(temp_err_neu1[j]*current.neu1[j]*(1-current.neu1[j]));
							for(int i=0;i<current.neu0.length;i++) {
								err_neu0[i] += delta_neu1[j]*syn0[j][i];
								temp_syn0[j][i] += parameters.alpha*delta_neu1[j]*current.neu0[i];
							}
						}
						
						// update word embedding
						for(int i=0;i<parameters.embeddingSize;i++) {
							E[current.wordID][i] -= err_neu0[i]+parameters.beta*E[current.wordID][i];
						}
							
						for(int j=0; j<temp_err_neu1.length; j++)
							temp_err_neu1[t] = err_neu0[j+current.neu0.length-current.neu1.length];
						
							
					}
					
					// update syn0
					for(int j=0;j<syn0.length;j++) {
						for(int i=0;i<syn0[0].length;i++) {
							syn0[j][i] -= temp_syn0[j][i]+parameters.beta*syn0[j][i];
						}
					}
					
					
					
				} else if(history.size()<parameters.bpttStep+1) { 
					// if the history of words we have processed is not enough, we do a normal BP
					History current = history.getLast();
					double[] delta_neu1 = new double[current.neu1.length];
					double[] err_neu0 = new double[current.neu0.length];
					for(int j=0;j<current.neu1.length;j++) {
						delta_neu1[j] = deltaTrunc(current.err_neu1[j]*current.neu1[j]*(1-current.neu1[j]));
						for(int i=0;i<current.neu0.length;i++) {
							err_neu0[i] += delta_neu1[j]*syn0[j][i];
							syn0[j][i] -= parameters.alpha*delta_neu1[j]*current.neu0[i]+parameters.beta*syn0[j][i];
						}
					}
					// update word embedding
					for(int i=0;i<parameters.embeddingSize;i++) {
						E[current.wordID][i] -= err_neu0[i]+parameters.beta*E[current.wordID][i];
					}
					
				} else {
					throw new Exception(); 
				}
				
				
			}
			
			// a sentence ends, refresh?  
			if(parameters.sentenceIndependent) {
				initHistory();
			}
			
		}
		
		System.out.println("loss: "+loss);
	}
}
