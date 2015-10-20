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
		filterWs = new ArrayList<>();
		eg2filterWs = new ArrayList<>();
		filterBs = new ArrayList<>();
		eg2filterBs = new ArrayList<>();
		Woes = new ArrayList<>();
		eg2Woes = new ArrayList<>();
		for(int k=0;k<parameters.filterNumber;k++) {
			/*
			 * We catenate all the embeddings in the window as the input of the filter and
			 * output 'filterMapNumber' nodes.
			 * As the filter is convolving in the sentence, each output node will form a line whose
			 * points correspond to the convolution steps.  
			 */
			double[][] filterW = new double[parameters.filterMapNumber][parameters.filterWindowSize[k]*parameters.embeddingSize];
			double[][] eg2filterW = new double[filterW.length][filterW[0].length];
			for(int i=0;i<filterW.length;i++) {
				for(int j=0;j<filterW[0].length;j++) {
					filterW[i][j] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
				}
			}
			filterWs.add(filterW);
			eg2filterWs.add(eg2filterW);
			double[] filterB = new double[parameters.filterMapNumber];
			double[] eg2filterB = new double[filterB.length];
			for(int i=0;i<filterB.length;i++) {
				filterB[i] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
			}
			filterBs.add(filterB);
			eg2filterBs.add(eg2filterB);
			
			double[][] Wo = new double[parameters.outputSize][parameters.filterMapNumber];
			double[][] eg2Wo = new double[Wo.length][Wo[0].length]; 
			for(int i=0;i<Wo.length;i++) {
				for(int j=0;j<Wo[0].length;j++) {
					Wo[i][j] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
				}
			}
			Woes.add(Wo);
			eg2Woes.add(eg2Wo);
			
			
		}
		
		Bo = new double[parameters.outputSize];
		for(int i=0;i<Bo.length;i++) {
			Bo[i] = random.nextDouble() * 2 * parameters.initRange - parameters.initRange;
		}
		eg2Bo = new double[Bo.length];
		
		eg2E = new double[E.length][E[0].length];
		
	}
	
	public double[] predict(Sentence ex) throws Exception {
		List<double[][]> Ses = new ArrayList<>();
		for(int filterIdx=0;filterIdx<parameters.filterNumber;filterIdx++) {
			// input -> convolutional map
			int timeToConvol = ex.tokens.size()-parameters.filterWindowSize[filterIdx]+1; // the times to convolute
			if(timeToConvol<=0)
				timeToConvol = 1; // sentence is less than window
			
			double[][] S = new double[timeToConvol][parameters.filterMapNumber]; // S[k][j]
			for(int k=0;k<timeToConvol;k++) { // k corresponds the begin position of each convolution
				
				int offset = 0;
				for(int wordCount=0;wordCount<parameters.filterWindowSize[filterIdx];wordCount++) {
					int wordIdx = k+wordCount;
					double[] emb = null;
					if(wordIdx<=ex.tokens.size()-1) {
						emb = E[ex.ids.get(wordIdx)];
					} else {
						emb = E[owner.getWordID(Parameters.PADDING)];
					}
					
					for(int j=0;j<parameters.filterMapNumber;j++) {
						for(int m=0;m<parameters.embeddingSize;m++) {
							S[k][j] += filterWs.get(filterIdx)[j][offset+m]*emb[m]; // W*Z
						}
					}
					offset += parameters.embeddingSize;
				}
				
				
				for(int j=0;j<parameters.filterMapNumber;j++) {
					S[k][j] += filterBs.get(filterIdx)[j];  // W*Z+B
					//u2[k][j] = S[k][j];
					S[k][j] = Util.sigmoid(S[k][j]); // activation
				}
				
				

			}
			
			
			Ses.add(S);

		}
		
		// S has been done, begin max-pooling to generate X
		List<double []> Xes = new ArrayList<>();
		List<int[]> maxRemembers = new ArrayList<>(); // record the max point in each column of S
		for(double[][] S:Ses) {
			double[] X = new double[parameters.filterMapNumber];
			int[] maxRemember = new int[X.length];
			Xes.add(X);
			maxRemembers.add(maxRemember);
			
			if(parameters.pooling==1) {
				for(int j=0;j<X.length;j++) {
					double sum = 0;
					for(int k=0;k<S.length;k++) {
						sum += S[k][j];
					}
					X[j] = sum/S.length;
				}
			} else if(parameters.pooling==2) {
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
		double[] Y = new double[parameters.outputSize];
		for(int i=0;i<Y.length;i++) {
			for(int filterIdx=0;filterIdx<parameters.filterNumber;filterIdx++) {
				for(int j=0;j<parameters.filterMapNumber;j++) {
					Y[i] += Woes.get(filterIdx)[i][j]*Xes.get(filterIdx)[j]; // Wo*X
				}
			}
			Y[i] += Bo[i]; // Wo*X+Bo
			Y[i] = Util.sigmoid(Y[i]); // sigmoid activation
		}
		
		return Y;
	}
	
	public GradientKeeper process(List<Sentence> examples) throws Exception {
		GradientKeeper keeper = new GradientKeeper(parameters, this);
		
		double loss = 0;
				
		for(Sentence ex:examples) {
			
			/*
			 *  send the sentence to each filter to convolute and each filter will generate a matrix
			 */
			List<double[][]> Ses = new ArrayList<>();
			List<double[][]> gradSes = new ArrayList<>();
			//List<double[][]> u2s = new ArrayList<>();
			for(int filterIdx=0;filterIdx<parameters.filterNumber;filterIdx++) {
				// input -> convolutional map
				int timeToConvol = ex.tokens.size()-parameters.filterWindowSize[filterIdx]+1; // the times to convolute
				if(timeToConvol<=0)
					timeToConvol = 1; // sentence is less than window
				
				double[][] S = new double[timeToConvol][parameters.filterMapNumber]; // S[k][j]
				double[][] gradS = new double[S.length][S[0].length];
				//double[][] u2 = new double[S.length][S[0].length];
				for(int k=0;k<timeToConvol;k++) { // k corresponds the begin position of each convolution
					
					int offset = 0;
					for(int wordCount=0;wordCount<parameters.filterWindowSize[filterIdx];wordCount++) {
						int wordIdx = k+wordCount;
						double[] emb = null;
						if(wordIdx<=ex.tokens.size()-1) {
							emb = E[ex.ids.get(wordIdx)];
						} else {
							emb = E[owner.getWordID(Parameters.PADDING)];
						}
						
						for(int j=0;j<parameters.filterMapNumber;j++) {
							for(int m=0;m<parameters.embeddingSize;m++) {
								S[k][j] += filterWs.get(filterIdx)[j][offset+m]*emb[m]; // W*Z
							}
						}
						offset += parameters.embeddingSize;
					}
					
					
					for(int j=0;j<parameters.filterMapNumber;j++) {
						S[k][j] += filterBs.get(filterIdx)[j];  // W*Z+B
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
				double[] X = new double[parameters.filterMapNumber];
				double[] gradX = new double[X.length];
				int[] maxRemember = new int[X.length];
				Xes.add(X);
				gradXes.add(gradX);
				maxRemembers.add(maxRemember);
				
				if(parameters.pooling==1) {
					for(int j=0;j<X.length;j++) {
						double sum = 0;
						for(int k=0;k<S.length;k++) {
							sum += S[k][j];
						}
						X[j] = sum/S.length;
					}
				} else if(parameters.pooling==2) {
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
			double[] Y = new double[parameters.outputSize];
			for(int i=0;i<Y.length;i++) {
				for(int filterIdx=0;filterIdx<parameters.filterNumber;filterIdx++) {
					for(int j=0;j<parameters.filterMapNumber;j++) {
						Y[i] += Woes.get(filterIdx)[i][j]*Xes.get(filterIdx)[j]; // Wo*X
					}
				}
				Y[i] += Bo[i]; // Wo*X+Bo
				Y[i] = Util.sigmoid(Y[i]); // sigmoid activation
			}
			
			
			loss += -(ex.polarity*Math.log(Y[0])+(1-ex.polarity)*Math.log(1-Y[0]))/examples.size();
			
			// Y -> X
			for(int i=0;i<parameters.outputSize;i++) {
				double delta1 = (Y[i]-ex.polarity)/examples.size();
				for(int filterIdx=0;filterIdx<parameters.filterNumber;filterIdx++) {
					for(int j=0;j<parameters.filterMapNumber;j++) {
						keeper.gradWoes.get(filterIdx)[i][j] += delta1*Xes.get(filterIdx)[j];
						gradXes.get(filterIdx)[j] += delta1*Woes.get(filterIdx)[i][j];
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
					if(parameters.pooling==1) {
						for(int k=0;k<S.length;k++) {
							gradS[k][j] = gradX[j];
						}
					} else if(parameters.pooling==2) {
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
					for(int wordCount=0;wordCount<parameters.filterWindowSize[sCount];wordCount++) {
						int wordIdx = k+wordCount;
						double[] emb = null;
						int embId = -1;
						if(wordIdx<=ex.tokens.size()-1) {
							embId = ex.ids.get(wordIdx);
						} else {
							embId = owner.getWordID(Parameters.PADDING);
						}
						emb = E[embId];

						for(int j=0;j<gradS[0].length;j++) {
							double delta2 = gradS[k][j]*S[k][j]*(1-S[k][j]);
							for(int m=0;m<parameters.embeddingSize;m++) {
								keeper.gradWs.get(sCount)[j][offset+m] += delta2*emb[m];
								keeper.gradE[embId][m] += delta2*filterWs.get(sCount)[j][offset+m];
							}
							keeper.gradBs.get(sCount)[j] += delta2;
						}
						offset += parameters.embeddingSize;
					}
				
				}
				
				
			}
			
	
		}
		
		// L2 Regularization
		for(int k=0;k<keeper.gradWs.size();k++) {
	    	double[][] gradW = keeper.gradWs.get(k);
		    for (int i = 0; i < gradW.length; ++i) {
		        for (int j = 0; j < gradW[i].length; ++j) {
		          loss += parameters.regParameter * filterWs.get(k)[i][j] * filterWs.get(k)[i][j] / 2.0;
		          gradW[i][j] += parameters.regParameter * filterWs.get(k)[i][j];
		        }
		      }
	    }
		
		for(int k=0;k<keeper.gradBs.size();k++) {
			double[] gradB = keeper.gradBs.get(k);
			for(int i=0; i < gradB.length; i++) {
				loss += parameters.regParameter * filterBs.get(k)[i] * filterBs.get(k)[i]/ 2.0;
				gradB[i] += parameters.regParameter * filterBs.get(k)[i];
			}
		}
		
		for(int k=0;k<keeper.gradWoes.size();k++) {
			double[][] gradWo = keeper.gradWoes.get(k);
			for(int i=0; i< gradWo.length; i++) {
				for(int j=0; j < gradWo[0].length;j++) {
					loss += parameters.regParameter * Woes.get(k)[i][j] * Woes.get(k)[i][j]/ 2.0;
					gradWo[i][j] += parameters.regParameter * Woes.get(k)[i][j];
				}
			}
		}
		
		for(int i=0;i<keeper.gradBo.length;i++) {
			loss += parameters.regParameter * Bo[i] * Bo[i]/ 2.0;
			keeper.gradBo[i] += parameters.regParameter * Bo[i];
		}
		
		for(int i=0; i< keeper.gradE.length; i++) {
			for(int j=0; j < keeper.gradE[0].length;j++) {
				loss += parameters.regParameter * E[i][j] * E[i][j]/ 2.0;
				keeper.gradE[i][j] += parameters.regParameter * E[i][j];
			}
		}
		
		if(debug)
			System.out.println("Cost = " + loss);
		
		return keeper;
	}
	
	public void updateWeights(GradientKeeper keeper) {
		// ada-gradient
		for(int k=0;k<filterWs.size();k++) {
	    	double[][] filterW = filterWs.get(k);
		    for (int i = 0; i < filterW.length; ++i) {
		        for (int j = 0; j < filterW[i].length; ++j) {
		          eg2filterWs.get(k)[i][j] += keeper.gradWs.get(k)[i][j] * keeper.gradWs.get(k)[i][j];
		          filterW[i][j] -= parameters.adaAlpha * keeper.gradWs.get(k)[i][j] / Math.sqrt(eg2filterWs.get(k)[i][j] + parameters.adaEps);
		        }
		      }
	    }
		
		for(int k=0;k<filterBs.size();k++) {
			double[] filterB = filterBs.get(k);
			for(int i=0; i < filterB.length; i++) {
				eg2filterBs.get(k)[i] += keeper.gradBs.get(k)[i] * keeper.gradBs.get(k)[i];
				filterB[i] -= parameters.adaAlpha * keeper.gradBs.get(k)[i] / Math.sqrt(eg2filterBs.get(k)[i] + parameters.adaEps); 
			}
		}
		
		for(int k=0;k<Woes.size();k++) {
			double[][] Wo = Woes.get(k);
			for(int i=0; i< Wo.length; i++) {
				for(int j=0; j < Wo[0].length;j++) {
					eg2Woes.get(k)[i][j] += keeper.gradWoes.get(k)[i][j] * keeper.gradWoes.get(k)[i][j];
					Wo[i][j] -= parameters.adaAlpha * keeper.gradWoes.get(k)[i][j] /  Math.sqrt(eg2Woes.get(k)[i][j] + parameters.adaEps);
				}
			}
		}
		
		for(int i=0;i<Bo.length;i++) {
			eg2Bo[i] += keeper.gradBo[i] * keeper.gradBo[i];
			Bo[i] -= parameters.adaAlpha * keeper.gradBo[i] / Math.sqrt(eg2Bo[i] + parameters.adaEps);
		}
		
		for(int i=0; i< E.length; i++) {
			for(int j=0; j < E[0].length;j++) {
				eg2E[i][j] += keeper.gradE[i][j] * keeper.gradE[i][j];
				E[i][j] -= parameters.adaAlpha * keeper.gradE[i][j] / Math.sqrt(eg2E[i][j] + parameters.adaEps);
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
		gradWs = new ArrayList<>();
		gradBs = new ArrayList<>();
		gradWoes = new ArrayList<>();
		for(int k=0;k<parameters.filterNumber;k++) {
			
			double[][] gradW = new double[cnn.filterWs.get(k).length][cnn.filterWs.get(k)[0].length];
			gradWs.add(gradW);
			
			double[] gradB = new double[cnn.filterBs.get(k).length];
			gradBs.add(gradB);
			
			double[][] gradWo = new double[cnn.Woes.get(k).length][cnn.Woes.get(k)[0].length];
			gradWoes.add(gradWo);
			
			
		}
		
		gradBo = new double[cnn.Bo.length];
		gradE = new double[cnn.E.length][cnn.E[0].length];
			
	}
}
