package cnn;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import common.Sentence;
import common.Util;


public class CNN implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6471262603387658363L;
	
	public Parameters parameters;
	public CNNSentence owner;
	
	public double[][] E;
	public List<double[][]> filterWs;
	public List<double[]> filterBs;
	public List<double[][]> Woes; // correspond to filter
	public double[] Bo;
	
	// eg* is for saving the gradient of the corresponding matrix
	public double[][] eg2E;
	public List<double[][]> eg2filterWs;
	public List<double[]> eg2filterBs;
	public List<double[][]> eg2Woes; 
	public double[] eg2Bo;
	
	public boolean debug;
	
	
	
	public CNN(Parameters parameters, CNNSentence owner, boolean debug, double[][] E) {
		this.parameters = parameters;
		this.owner = owner;
		this.debug = debug;
		this.E = E;
		
		Random random = new Random(System.currentTimeMillis());
		this.filterWs = new ArrayList<>();
		this.eg2filterWs = new ArrayList<>();
		this.filterBs = new ArrayList<>();
		this.eg2filterBs = new ArrayList<>();
		this.Woes = new ArrayList<>();
		this.eg2Woes = new ArrayList<>();
		for(int k=0;k<parameters.filterNumber;k++) {
			/*
			 * We catenate all the embeddings in the window as the input of the filter and
			 * output 'filterMapNumber' nodes.
			 * As the filter is convolving in the sentence, each output node will form a line whose
			 * points correspond to the convolution steps.  
			 */
			
			// 1: filterW[300][3*50]
			// 2: filterW[300][4*50]
			// 2: filterW[300][5*50]
			double[][] filterW = new double[parameters.filterMapNumber][parameters.filterWindowSize[k]*parameters.embeddingSize];
			double[][] eg2filterW = new double[filterW.length][filterW[0].length];
			for(int i=0;i<filterW.length;i++) {
				for(int j=0;j<filterW[0].length;j++) {
					filterW[i][j] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
				}
			}
			this.filterWs.add(filterW);
			this.eg2filterWs.add(eg2filterW);
			double[] filterB = new double[parameters.filterMapNumber];
			double[] eg2filterB = new double[filterB.length];
			for(int i=0;i<filterB.length;i++) {
				filterB[i] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
			}
			this.filterBs.add(filterB);
			this.eg2filterBs.add(eg2filterB);
			
			// Wo: [1][300]
			double[][] Wo = new double[parameters.outputSize][parameters.filterMapNumber];
			double[][] eg2Wo = new double[Wo.length][Wo[0].length]; 
			for(int i=0;i<Wo.length;i++) {
				for(int j=0;j<Wo[0].length;j++) {
					Wo[i][j] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
				}
			}
			this.Woes.add(Wo);
			this.eg2Woes.add(eg2Wo);
			
			
		}
		
		this.Bo = new double[parameters.outputSize];
		for(int i=0;i<this.Bo.length;i++) {
			this.Bo[i] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
		}
		this.eg2Bo = new double[this.Bo.length];
		
		this.eg2E = new double[E.length][E[0].length];
		
	}
	
	public double[] predict(Sentence ex) throws Exception {
		List<double[][]> Ses = new ArrayList<>();
		for(int filterIdx=0;filterIdx<this.parameters.filterNumber;filterIdx++) {
			// input -> convolutional map
			int timeToConvol = ex.tokens.size()-this.parameters.filterWindowSize[filterIdx]+1; // the times to convolute
			if(timeToConvol<=0)
				timeToConvol = 1; // sentence is less than window
			
			double[][] S = new double[timeToConvol][this.parameters.filterMapNumber]; // S[k][j]
			for(int k=0;k<timeToConvol;k++) { // k corresponds the begin position of each convolution
				
				int offset = 0;
				for(int wordCount=0;wordCount<this.parameters.filterWindowSize[filterIdx];wordCount++) {
					int wordIdx = k+wordCount;
					double[] emb = null;
					if(wordIdx<=ex.tokens.size()-1) {
						emb = this.E[ex.ids.get(wordIdx)];
					} else {
						emb = this.E[this.owner.getWordID(Parameters.PADDING)];
					}
					
					for(int j=0;j<this.parameters.filterMapNumber;j++) {
						for(int m=0;m<this.parameters.embeddingSize;m++) {
							// k是以某个长度(3)来做卷积的第k行，每一次都把3个单词(50维向量)卷成一个300维的向量
							S[k][j] += this.filterWs.get(filterIdx)[j][offset+m]*emb[m]; // W*Z
						}
					}
					offset += this.parameters.embeddingSize;
				}
				
				
				for(int j=0;j<this.parameters.filterMapNumber;j++) {
					S[k][j] += this.filterBs.get(filterIdx)[j];  // W*Z+B
					//u2[k][j] = S[k][j];
					S[k][j] = Util.sigmoid(S[k][j]); // activation
				}
				
				

			}
			
			
			Ses.add(S); // Ses.size() == this.parameters.filterNumber 

		}
		
		// S has been done, begin max-pooling to generate X
		List<double []> Xes = new ArrayList<>();
		List<int[]> maxRemembers = new ArrayList<>(); // record the max point in each column of S
		for(double[][] S:Ses) {
			double[] X = new double[this.parameters.filterMapNumber];
			int[] maxRemember = new int[X.length];
			Xes.add(X);
			maxRemembers.add(maxRemember);
			
			// X -> [1, 300]
			if(this.parameters.pooling==1) {
				for(int j=0;j<X.length;j++) {
					double sum = 0;
					for(int k=0;k<S.length;k++) {
						sum += S[k][j];
					}
					X[j] = sum/S.length;
				}
			} else if(this.parameters.pooling==2) {
				for(int j=0;j<X.length;j++) {
					double max = S[0][j];
					int maxK = 0;
					for(int k=1;k<S.length;k++) {
						if(S[k][j]>max) {
							max = S[k][j];
							maxK = k;
						}
					}
					X[j] = max;

					maxRemember[j] = maxK;
				}
			} else
				throw new Exception();
			
			
		}
		
		// X -> Y, a logistic regression
		double[] Y = new double[this.parameters.outputSize];
		for(int i=0;i<Y.length;i++) {
			for(int filterIdx=0;filterIdx<this.parameters.filterNumber;filterIdx++) {
				for(int j=0;j<this.parameters.filterMapNumber;j++) {
					Y[i] += this.Woes.get(filterIdx)[i][j]*Xes.get(filterIdx)[j]; // Wo*X
				}
			}
			Y[i] += this.Bo[i]; // Wo*X+Bo
			Y[i] = Util.sigmoid(Y[i]); // sigmoid activation
		}
		
		return Y;
	}
	
	public GradientKeeper process(List<Sentence> examples) throws Exception {
		GradientKeeper keeper = new GradientKeeper(this.parameters, this);
		
		double loss = 0;
				
		for(Sentence ex:examples) {
			
			/*
			 *  send the sentence to each filter to convolute and each filter will generate a matrix
			 */
			List<double[][]> Ses = new ArrayList<>();
			List<double[][]> gradSes = new ArrayList<>();
			//List<double[][]> u2s = new ArrayList<>();
			for(int filterIdx=0;filterIdx<this.parameters.filterNumber;filterIdx++) {
				// input -> convolutional map
				int timeToConvol = ex.tokens.size()-this.parameters.filterWindowSize[filterIdx]+1; // the times to convolute
				if(timeToConvol<=0)
					timeToConvol = 1; // sentence is less than window
				
				double[][] S = new double[timeToConvol][this.parameters.filterMapNumber]; // S[k][j]
				double[][] gradS = new double[S.length][S[0].length];
				//double[][] u2 = new double[S.length][S[0].length];
				for(int k=0;k<timeToConvol;k++) { // k corresponds the begin position of each convolution
					
					int offset = 0;
					for(int wordCount=0;wordCount<this.parameters.filterWindowSize[filterIdx];wordCount++) {
						int wordIdx = k+wordCount;
						double[] emb = null;
						if(wordIdx<=ex.tokens.size()-1) {
							emb = this.E[ex.ids.get(wordIdx)];
						} else {
							emb = this.E[this.owner.getWordID(Parameters.PADDING)];
						}
						
						for(int j=0;j<this.parameters.filterMapNumber;j++) {
							for(int m=0;m<this.parameters.embeddingSize;m++) {
								S[k][j] += this.filterWs.get(filterIdx)[j][offset+m]*emb[m]; // W*Z
							}
						}
						offset += this.parameters.embeddingSize;
					}
					
					
					for(int j=0;j<this.parameters.filterMapNumber;j++) {
						S[k][j] += this.filterBs.get(filterIdx)[j];  // W*Z+B
						//u2[k][j] = S[k][j];
						S[k][j] = Util.sigmoid(S[k][j]); // activation
					}
					
					

				}
				
				
				Ses.add(S);
				gradSes.add(gradS);
				//u2s.add(u2);
			}
			
			// S has been done, begin max-pooling to generate X
			List<double []> Xes = new ArrayList<>();
			List<double []> gradXes = new ArrayList<>();
			List<int[]> maxRemembers = new ArrayList<>(); // record the max point in each column of S
			for(double[][] S:Ses) {
				double[] X = new double[this.parameters.filterMapNumber];
				double[] gradX = new double[X.length];
				int[] maxRemember = new int[X.length];
				Xes.add(X);
				gradXes.add(gradX);
				maxRemembers.add(maxRemember);
				
				if(this.parameters.pooling==1) { //avg pooling
					for(int j=0;j<X.length;j++) {
						double sum = 0;
						for(int k=0;k<S.length;k++) {
							sum += S[k][j];
						}
						X[j] = sum/S.length;
					}
				} else if(this.parameters.pooling==2) { // max pooling
					for(int j=0;j<X.length;j++) {
						double max = S[0][j];
						int maxK = 0;
						for(int k=1;k<S.length;k++) {
							if(S[k][j]>max) {
								max = S[k][j];
								maxK = k;
							}
						}
						X[j] = max;
	
						maxRemember[j] = maxK;
					}
				} else
					throw new Exception();
				
				
			}
			
			// X -> Y, a logistic regression
			double[] Y = new double[this.parameters.outputSize];
			for(int i=0;i<Y.length;i++) {
				for(int filterIdx=0;filterIdx<this.parameters.filterNumber;filterIdx++) {
					for(int j=0;j<this.parameters.filterMapNumber;j++) {
						Y[i] += this.Woes.get(filterIdx)[i][j]*Xes.get(filterIdx)[j]; // Wo*X
					}
				}
				Y[i] += this.Bo[i]; // Wo*X+Bo
				Y[i] = Util.sigmoid(Y[i]); // sigmoid activation
			}
			
			// calculate the loss
			for (int i=0;i<this.parameters.outputSize;i++) {
				loss += -(ex.category[i]*Math.log(Y[i])+(1-ex.category[i])*Math.log(1-Y[i]))/examples.size();
			}
			
			
			// Y -> X
			for(int i=0;i<this.parameters.outputSize;i++) {
				double delta1 = (Y[i]-ex.category[i])/examples.size();
				for(int filterIdx=0;filterIdx<this.parameters.filterNumber;filterIdx++) {
					for(int j=0;j<this.parameters.filterMapNumber;j++) {
						keeper.gradWoes.get(filterIdx)[i][j] += delta1*Xes.get(filterIdx)[j];
						gradXes.get(filterIdx)[j] += delta1*this.Woes.get(filterIdx)[i][j];
					}
				}
				keeper.gradBo[i] += delta1;
			}
			
			// X -> S
			for(int sCount=0;sCount<Ses.size();sCount++) {
				double[][] S = Ses.get(sCount);
				double[][] gradS = gradSes.get(sCount);
				int[] maxRemember = maxRemembers.get(sCount);
				double[] gradX = gradXes.get(sCount);
				for(int j=0;j<S[0].length;j++) {
					if(this.parameters.pooling==1) {
						for(int k=0;k<S.length;k++) {
							gradS[k][j] = gradX[j];
						}
					} else if(this.parameters.pooling==2) {
						for(int k=0;k<S.length;k++) {
							if(maxRemember[j]==k)
								gradS[k][j] = gradX[j];
							else
								gradS[k][j] = 0;
						}
					} else
						throw new Exception();
					
					
				}
				
			}
			
			// S -> Z
			for(int sCount=0;sCount<Ses.size();sCount++) {
				double[][] gradS = gradSes.get(sCount);
				//double[][] u2 = u2s.get(sCount); 
				double[][] S = Ses.get(sCount);
				for(int k=0;k<gradS.length;k++) {
					
					int offset = 0;
					for(int wordCount=0;wordCount<this.parameters.filterWindowSize[sCount];wordCount++) {
						int wordIdx = k+wordCount;
						double[] emb = null;
						int embId = -1;
						if(wordIdx<=ex.tokens.size()-1) {
							embId = ex.ids.get(wordIdx);
						} else {
							embId = this.owner.getWordID(Parameters.PADDING);
						}
						emb = this.E[embId];

						for(int j=0;j<gradS[0].length;j++) {
							double delta2 = gradS[k][j]*S[k][j]*(1-S[k][j]);
							for(int m=0;m<this.parameters.embeddingSize;m++) {
								keeper.gradWs.get(sCount)[j][offset+m] += delta2*emb[m];
								keeper.gradE[embId][m] += delta2*this.filterWs.get(sCount)[j][offset+m];
							}
							keeper.gradBs.get(sCount)[j] += delta2;
						}
						offset += this.parameters.embeddingSize;
					}
				
				}
				
				
			}
			
	
		}
		
		// L2 Regularization
		for(int k=0;k<keeper.gradWs.size();k++) {
	    	double[][] gradW = keeper.gradWs.get(k);
		    for (int i = 0; i < gradW.length; ++i) {
		        for (int j = 0; j < gradW[i].length; ++j) {
		          loss += this.parameters.regParameter * this.filterWs.get(k)[i][j] * this.filterWs.get(k)[i][j] / 2.0;
		          gradW[i][j] += this.parameters.regParameter * this.filterWs.get(k)[i][j];
		        }
		      }
	    }
		
		for(int k=0;k<keeper.gradBs.size();k++) {
			double[] gradB = keeper.gradBs.get(k);
			for(int i=0; i < gradB.length; i++) {
				loss += this.parameters.regParameter * this.filterBs.get(k)[i] * this.filterBs.get(k)[i]/ 2.0;
				gradB[i] += this.parameters.regParameter * this.filterBs.get(k)[i];
			}
		}
		
		for(int k=0;k<keeper.gradWoes.size();k++) {
			double[][] gradWo = keeper.gradWoes.get(k);
			for(int i=0; i< gradWo.length; i++) {
				for(int j=0; j < gradWo[0].length;j++) {
					loss += this.parameters.regParameter * this.Woes.get(k)[i][j] * this.Woes.get(k)[i][j]/ 2.0;
					gradWo[i][j] += this.parameters.regParameter * this.Woes.get(k)[i][j];
				}
			}
		}
		
		for(int i=0;i<keeper.gradBo.length;i++) {
			loss += this.parameters.regParameter * this.Bo[i] * this.Bo[i]/ 2.0;
			keeper.gradBo[i] += this.parameters.regParameter * this.Bo[i];
		}
		
		for(int i=0; i< keeper.gradE.length; i++) {
			for(int j=0; j < keeper.gradE[0].length;j++) {
				loss += this.parameters.regParameter * this.E[i][j] * this.E[i][j]/ 2.0;
				keeper.gradE[i][j] += this.parameters.regParameter * this.E[i][j];
			}
		}
		
		if(this.debug)
			System.out.println("Cost = " + loss);
		
		return keeper;
	}
	
	public void updateWeights(GradientKeeper keeper) {
		// ada-gradient
		for(int k=0;k<this.filterWs.size();k++) {
	    	double[][] filterW = this.filterWs.get(k);
		    for (int i = 0; i < filterW.length; ++i) {
		        for (int j = 0; j < filterW[i].length; ++j) {
		          this.eg2filterWs.get(k)[i][j] += keeper.gradWs.get(k)[i][j] * keeper.gradWs.get(k)[i][j];
		          filterW[i][j] -= this.parameters.adaAlpha * keeper.gradWs.get(k)[i][j] / Math.sqrt(this.eg2filterWs.get(k)[i][j] + this.parameters.adaEps);
		        }
		      }
	    }
		
		for(int k=0;k<this.filterBs.size();k++) {
			double[] filterB = this.filterBs.get(k);
			for(int i=0; i < filterB.length; i++) {
				this.eg2filterBs.get(k)[i] += keeper.gradBs.get(k)[i] * keeper.gradBs.get(k)[i];
				filterB[i] -= this.parameters.adaAlpha * keeper.gradBs.get(k)[i] / Math.sqrt(this.eg2filterBs.get(k)[i] + this.parameters.adaEps); 
			}
		}
		
		for(int k=0;k<this.Woes.size();k++) {
			double[][] Wo = this.Woes.get(k);
			for(int i=0; i< Wo.length; i++) {
				for(int j=0; j < Wo[0].length;j++) {
					this.eg2Woes.get(k)[i][j] += keeper.gradWoes.get(k)[i][j] * keeper.gradWoes.get(k)[i][j];
					Wo[i][j] -= this.parameters.adaAlpha * keeper.gradWoes.get(k)[i][j] /  Math.sqrt(this.eg2Woes.get(k)[i][j] + this.parameters.adaEps);
				}
			}
		}
		
		for(int i=0;i<this.Bo.length;i++) {
			this.eg2Bo[i] += keeper.gradBo[i] * keeper.gradBo[i];
			this.Bo[i] -= this.parameters.adaAlpha * keeper.gradBo[i] / Math.sqrt(this.eg2Bo[i] + this.parameters.adaEps);
		}
		
		for(int i=0; i< this.E.length; i++) {
			for(int j=0; j < this.E[0].length;j++) {
				this.eg2E[i][j] += keeper.gradE[i][j] * keeper.gradE[i][j];
				this.E[i][j] -= this.parameters.adaAlpha * keeper.gradE[i][j] / Math.sqrt(this.eg2E[i][j] + this.parameters.adaEps);
			}
		}
		
		

		
	}
}

class GradientKeeper {
	public List<double[][]> gradWs;
	public List<double[]> gradBs;
	public List<double[][]> gradWoes;
	public double[] gradBo;
	public double[][] gradE;
	
	
	// initialize gradient matrixes, their dimensions are identical to the corresponding matrixes.
	public GradientKeeper(Parameters parameters, CNN cnn) {
		this.gradWs = new ArrayList<>();
		this.gradBs = new ArrayList<>();
		this.gradWoes = new ArrayList<>();
		for(int k=0;k<parameters.filterNumber;k++) {
			
			double[][] gradW = new double[cnn.filterWs.get(k).length][cnn.filterWs.get(k)[0].length];
			this.gradWs.add(gradW);
			
			double[] gradB = new double[cnn.filterBs.get(k).length];
			this.gradBs.add(gradB);
			
			double[][] gradWo = new double[cnn.Woes.get(k).length][cnn.Woes.get(k)[0].length];
			this.gradWoes.add(gradWo);
			
		}
		
		this.gradBo = new double[cnn.Bo.length];
		this.gradE = new double[cnn.E.length][cnn.E[0].length];
			
	}
}
