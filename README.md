# MITACs_TQI


### Synonym_Counting
Required additional documents: Arxiv Articles => https://www.kaggle.com/datasets/Cornell-University/arxiv

inputs: 
  1) .csv of 2-tier taxonomy
  2) .json of texts (with publication date)

output:
  1) Count of each article type by month

### Word2Vec_Classifier
Required additional documents: GoogleNews-vectors-negative300.bin => https://www.kaggle.com/datasets/sandreds/googlenewsvectorsnegative300

inputs: 
  1) .csv of known NE-Classification pairs
  2) .bin archive of pre-trained word vectors

output:
  1) Naive Baye's model which classifies NE's 

### Data_Augmentation
