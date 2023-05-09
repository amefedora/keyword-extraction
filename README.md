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
- Drop columns
  This experiment will only use the Id, Year, and the combination of Title and Abstract attribute.
- Drop missing values
  Since, there is a lot of missing abstract in dataset I decide to drop all the records with missing abstract.
  ![1](https://github.com/amefedora/keyword-extraction/assets/65814424/a2de6b9a-e51a-4ba0-8d81-1797aa94acf7)

- 
