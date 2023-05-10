# Keyword Extraction Visualization Experiment

Commonly, keyword is the most important component to represent the content of an article. Besides, keyword also has an important role to help us find an article through search engine, bibliographic database, etc. Keyword that we usually found in an article is result of the article extraction. This experiment aim to extract keywords from abstract of an article with Natural Language Processing. The extracted keywords then will visualized using Python Chart. 

# Tools
Programming Language: Python

# Dataset Description 
papers.csv have 7241 records with 7 attributes of Neural Information Processing Systems (NIPS) papers.

| Attribute/Column  | Description |
| ------------- | ------------- |
| Id | ------------- |
| Year | ------------- |
| Title | ------------- |
| Event Type | ------------- |
| Year | ------------- |
| PDF Name | ------------- |
| Abstract | ------------- |
| Paper Text | ------------- |

# Work Flow
1. Data Preparation
2. Text Processing
3. Exploratory Data Analysis (EDA)
4. N-Gram Vectorizer
5. Keyword Extraction with TF-IDF


# Data Preparation
- Library import
  - Pandas
  - Re
  - NLTK
  - Sklear
  - PIL
  - Wordcloud
  - Matplotlib
  - Seaborn
  - Scipy
  
- Drop missing values
  Since, there is a lot of missing abstract in dataset I decide to drop all the records with missing abstract and end up only 3924 records left.
  
  ![1](https://github.com/amefedora/keyword-extraction/assets/65814424/a2de6b9a-e51a-4ba0-8d81-1797aa94acf7)
  
- Drop unused columns and combine the Title and Abstract attribute into one record
  This experiment will only use the Id, Year, and the combination of Title and Abstract attribute therefore I drop the other unuse attributes. The combination of Title and Abstract attribute then called as **abstract1** attribute.
  
  ![2](https://github.com/amefedora/keyword-extraction/assets/65814424/45114618-2585-43ee-8276-1ec9415c2df4)

# Text Processing
1. Remove the punctuation, tags, special characters and digit of abstract1 attribute and convert all words to lowercase.
2. Create abstract1 word count column as new attribute. The word count column will filled with the value of word count. 
  ![3](https://github.com/amefedora/keyword-extraction/assets/65814424/ac5cefc9-982b-475e-809b-e6d06e5da6e4)
3. Create a new variable to classified the common and uncommon word from the abstract1 column. In this experiment, if the word count is more than 20 then the word classified as the common word, otherwise if the word count is less than 20 then the word classified as the uncommon word. The common word and the uncommon word will classification will use to make a stop word corpus later. 
  
  | Common Words  | Uncommon Words |
  | ------------- | ------------- |
  | ![4](https://github.com/amefedora/keyword-extraction/assets/65814424/12a35422-9fad-476f-8dbc-5e68af3b87ec) |  ![5](https://github.com/amefedora/keyword-extraction/assets/65814424/82ef7cb8-6eda-4e7d-83aa-610a34ba65b0)|
  


