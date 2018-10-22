Welcome to the Author-Identification wiki!

Task- Identify the Author of an article. 
Implementing paper- deep learning based authorship identification
Refer https://web.stanford.edu/class/cs224n/reports/2760185.pdf more more details.

Dataset- Reuter_50_50 Approach- Used LSTM and GRU units to compare their performance. Glove representation used for the article.

Approach-
1)Sentence level : 
Every sentence is divided into words and then these word glove representation is passed to LSTM/GRU unit.
2)Article level : 
Every article is divided into sentence. Each sentence is represented as mean of the glove representation of each word. Then the article is passed to LSTM/GRU unit.
