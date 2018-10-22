Welcome to the Author-Identification wiki!

Task- Identify the Author of an article Dataset- Reuter_50_50 Approach- Used LSTM and GRU units to compare their performance. Glove representation used for the article.

Sentence level Every sentence is divided into words and then these word glove representation is passed to LSTM/GRU unit.
Article level Every article is divided into sentence. Each sentence is represented as mean of the glove representation of each word. Then the article is passed to LSTM/GRU unit.
