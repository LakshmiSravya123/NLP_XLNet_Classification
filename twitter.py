
######################################################BERT gender_twiiter###################################################


! pip install transformers
% pip install sentencepiece

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import AdamW

from tqdm import trange
import torch
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('wordnet')
stopwords = ['i','age','lol','rofl','haha','hehe','Mother','daughter','elder','grandfather','son','father','grandmother','uncle','Aunt','me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't","young","younger","old","older","student","class","girl","boy","yay","can","can't"]

data = pd.read_csv("/content/drive/My Drive/Colab Notebooks/gender-classifier-DFE-791531.csv",encoding="latin1")
data = pd.concat([data.gender,data.description,data.text],axis=1)
data=data[:30000]
#make some preperation for prediction, first change male->0 female ->1
data.gender = [1 if each =="female" else 0 for each in data.gender]

data.dropna(axis=0,inplace=True)
data.head(5)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
n_gpu = torch.cuda.device_count()

#Now prepare text of description data for prediction. Like, making lowercase, omitting unnecessary words,stopping words etc. 
description_list = []
lemma = nltk.WordNetLemmatizer()
data["des"] = data["text"].map(str) + data["description"] 
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True, cutoffs=[])
tokenized_texts = [tokenizer.tokenize(sent) for sent in data.des]
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=100, dtype="long", truncating="post", padding="post")
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, data.iloc[:,0].values, 
                                                            random_state=56, test_size=0.2)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=56, test_size=0.2)
"""
clf = MultinomialNB(alpha=41).fit(train_inputs,train_labels)
prediction=clf.predict(validation_masks)
print("multinobial training: ",clf.score(train_inputs,train_labels))
print("multinobial accuracy: ",clf.score(validation_masks,validation_labels))
"""
import os
directory_path = os.getcwd()
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)
batch_size = 16

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)
model.cuda()
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters,
                     lr=2e-5)
train_loss_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 100

validation_accuracy = float("-inf")
training_accuracy = float("-inf")

# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):
  
  
  # Training
  
  # Set our model to training mode (as opposed to evaluation mode)
  model.train()
  
  # Tracking variables
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  
  # Train the data for one epoch
  for step, batch in enumerate(train_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Clear out the gradients (by default they accumulate)
    optimizer.zero_grad()
    # Forward pass
    outputs = model(b_input_ids.long(), token_type_ids=None, attention_mask=b_input_mask, labels=b_labels.cuda())
    loss = outputs[0]
    logits = outputs[1]
    train_loss_set.append(loss.item())    
    # Backward pass
    loss.backward()
    # Update parameters and take a step using the computed gradient
    optimizer.step()
    
    
    # Update tracking variables
    tr_loss += loss.item()
    nb_tr_examples += b_input_ids.size(0)
    nb_tr_steps += 1
    
  print("Training loss", tr_loss)
  with torch.no_grad():
    correct = 0
    total = 0
    for i, batch in enumerate(validation_dataloader):
      batch = tuple(t.to(device) for t in batch)
      # Unpack the inputs from our dataloader
      b_input_ids, b_input_mask, b_labels = batch
      # Forward pass
      outputs = model(b_input_ids.long(), token_type_ids=None, attention_mask=b_input_mask)
      # print (outputs)
      prediction = torch.argmax(outputs[0],dim=1)
      total += b_labels.size(0)
      correct+=(prediction==b_labels).sum().item()
    if((100 * (correct / total)) > validation_accuracy):
        validation_accuracy = 100 * correct / total
        print('saving best model: {} %'.format(validation_accuracy))
        torch.save(model.state_dict(), directory_path+'/model_with_language_model.ckpt')
# Test the model

print('Test Accuracy of the finetuned model on val data is: {} %'.format(100 * correct / total))
