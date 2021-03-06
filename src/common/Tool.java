package common;


import java.util.regex.Pattern;

import cn.fox.biomedical.Dictionary;
import cn.fox.biomedical.Sider;
import cn.fox.machine_learning.BrownCluster;
import cn.fox.machine_learning.Perceptron;
import cn.fox.nlp.SentenceSplitter;
import cn.fox.stanford.Tokenizer;
import cn.fox.utils.StopWord;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.Morphology;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;

public class Tool {
	public SentenceSplitter sentSplit;
	public TokenizerFactory<CoreLabel> tokenizerFactory;
	public Tokenizer tokenizer;
	public MaxentTagger tagger;
	//public BiocXmlParser xmlParser;
	public Morphology morphology;
	public LexicalizedParser lp;
	public GrammaticalStructureFactory gsf;

	
	public Dictionary drugbank;
	public Dictionary jochem;
	public Dictionary ctdchem;
	
	public Dictionary humando;
	public Dictionary ctdmedic;

	
	public Dictionary chemElem;
	
	public Pattern complexNounPattern;
	
	public Sider sider;

	public BrownCluster entityBC;
	public BrownCluster relationBC;
	
	public StopWord stopWord;
}
