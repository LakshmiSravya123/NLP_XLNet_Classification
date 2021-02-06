# NLP_XLNet_Classification


# Datsets Used
- gender.csv (blog data set)
http://www.cs.uic.edu/~liub/FBS/blog-gender-dataset.rar
- gender-classifier-DFE-791531 (twitter data set)
https://www.kaggle.com/crowdflower/twitter-user-gender-classification



# NLP  Classifier
XLNet is an auto-regressive language model that outputs the joint probability of a sequence of tokens based on the transformer architecture with recurrence. XLNet main training objective is learning conditional distributions for all permutations of tokens in sequence. BERT model improves state of the art by incorporating left and right contexts into predictions, whereas XLNet contains a combination of other words in a sequence for predictions. The method proposed by XLNet is called "Permutation Language Modelling." XLNet seems to abort advantage from BERT, GPT-2, and Transformer-XL as it uses permutation language modeling to learn both side information (from BERT). It handles long text sequence by using a hidden state (from Transformer-XL). XLNet is “autoregressive,” where BERT is an “auto encoder.” Autoregressive models are better at generating new text.

# Methodology 
Train-Test split: The training set is divided into 80% of training data and 20% as a test data set. After training, the whole test data is passed to the classifier for predictions.

# Coding imports and installations
- Tensorflow
- Transformers
- Torch
- nltk
- pandas
- keras
- sentencepiece

# Gender Classification Results 
 ## Twitter dataset
Accuracy on Language Model (XLNet): 75.5% much better than any of the other ML classifiers 

## Blog data set
Accuracy on Language Model (XLNet): 71.6% way better than the ML classfiers



# Languages
Pythong (NLP XLNet)

# References
- https://github.com/Shivampanwar/Bert-text-classification/blob/master/XLNet/xlnet_experimentation.ipynb

