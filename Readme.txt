Author: Li Fei
Date: October 11th, 2015
Email: lifei_csnlp@whu.edu.cn

This is a java project using Convolutional Neutral Network to classify the sentiment of a comment sentence. The idea and implements are
mostly similar with that of Kim's paper in EMNLP 2014. 
However, there are some slight changes between mine and Kim's. Please see the code for details.
The data are the same as those of Kim's paper which are provided by Pang and Lee in ACL 2005.
In the current configuration, the accuracy can achieve 73% approximately.

 
The relevant papers you need to read.
author    = {Kim, Yoon},
title     = {Convolutional Neural Networks for Sentence Classification},

author    = {Pang, Bo  and  Lee, Lillian},
title     = {Seeing Stars: Exploiting Class Relationships for Sentiment Categorization with Respect to Rating Scales},


Data: October 20th, 2015
I have added a recurrent neural network to classify a sentence. In the current configuration, the accuracy can achieve 70% approximately.
All the ideas and code are reference Mikolov's doctor thesis.

Mikolov Tomas: Statistical Language Models based on Neural Networks. PhD thesis, Brno University of Technology, 2012.
