# Viva Questions and Answers  
## Assignment No. 5  
### Title: Implementing the Continuous Bag of Words (CBOW) Model  

---

### **1. What is NLP?**
Natural Language Processing (NLP) is a field of Artificial Intelligence that focuses on enabling computers to understand, interpret, and generate human language. It combines linguistics, computer science, and machine learning to process text and speech data.

---

### **2. What is Word Embedding in NLP?**
Word embedding is a technique used to represent words as dense numerical vectors in a continuous vector space. Words with similar meanings have similar vector representations, allowing the model to capture semantic relationships between them.

---

### **3. Explain Word2Vec techniques.**
**Word2Vec** is a popular word embedding technique that transforms words into vectors using neural networks.  
It has two main architectures:
- **CBOW (Continuous Bag of Words):** Predicts a target word based on its surrounding context words.  
- **Skip-Gram:** Predicts context words based on a given target word.  

Both methods help capture semantic and syntactic relationships between words.

---

### **4. What are some applications of Word Embeddings in NLP?**
- Sentiment analysis  
- Machine translation  
- Text classification  
- Chatbots and conversational AI  
- Document similarity detection  
- Named entity recognition (NER)

---

### **5. Explain CBOW architecture.**
The **Continuous Bag of Words (CBOW)** model predicts a target word using the context of surrounding words.  
It consists of:
1. **Input Layer:** Takes multiple context words as input.  
2. **Projection/Embedding Layer:** Converts each context word into its embedding vector.  
3. **Hidden Layer:** Averages context embeddings to form a context representation.  
4. **Output Layer:** Predicts the target word based on this representation.

---

### **6. What are the inputs and outputs of the CBOW model?**
- **Input:** A group of surrounding words (context).  
- **Output:** The central target word that the model tries to predict.

For example:  
Sentence – “The cat sits on the mat”  
If the target word is “sits” and window size = 2,  
Input → [“The”, “cat”, “on”, “the”]  
Output → [“sits”]

---

### **7. What is a Tokenizer?**
A **Tokenizer** is a preprocessing tool that splits text into individual tokens (words or subwords). It helps convert sentences into numerical data suitable for machine learning models.

---

### **8. Explain the window size parameter in detail for CBOW.**
The **window size** determines how many words on each side of the target word are used as context.  
For example, if the window size is 2 in the sentence *“The cat sits on the mat”*, then for the target word “sits”, the context words are [“The”, “cat”, “on”, “the”].  
A larger window size captures more global context, while a smaller window captures local context.

---

### **9. Explain the Embedding and Lambda layers from Keras.**
- **Embedding Layer:**  
  Converts integer-encoded words into dense vectors of fixed size. It’s used to learn word representations during training.  
  Example: Converts word indices into 100-dimensional vectors.

- **Lambda Layer:**  
  Used to perform custom operations within the neural network (e.g., averaging embeddings or computing mathematical operations without trainable parameters).

---

### **10. What is the purpose of yield() in Python?**
The `yield()` keyword is used in generators to produce a sequence of results lazily (one at a time). It allows a function to return data iteratively instead of computing everything at once, saving memory and improving performance during data generation (e.g., in training data batches).

---

### **11. What is the main difference between CBOW and Skip-Gram?**

| Feature | CBOW | Skip-Gram |
|----------|-------|-----------|
| **Objective** | Predicts target word from context | Predicts context words from target word |
| **Computation** | Faster, efficient for large datasets | Slower, works better with smaller datasets |
| **Performance** | Better for frequent words | Better for rare words |

---

### **12. What is the advantage of using CBOW?**
- It is computationally efficient.  
- Works well for large datasets.  
- Captures semantic meaning of words effectively through context-based prediction.

---

### **13. What is the role of Gensim in NLP?**
**Gensim** is a Python library used for topic modeling and word embedding. It provides implementations of Word2Vec, Doc2Vec, and other models for text representation and semantic analysis.

---

### **14. What is the importance of word vectors in NLP?**
Word vectors help machines understand word meaning and relationships by mapping them into a vector space where similar words are positioned closer together (e.g., “king” – “man” + “woman” ≈ “queen”).

---

### **15. What are some real-world applications of the CBOW model?**
- Search engines (context-based suggestions)  
- Machine translation  
- Speech-to-text systems  
- Text summarization  
- Chatbots and virtual assistants  

---

### **16. What is one-hot encoding and why is it used in NLP?**
**One-hot encoding** represents words as binary vectors where only one position (corresponding to the word index) is set to 1, and all others are 0. It’s used to uniquely identify words before converting them into dense embeddings.

---

### **17. What is the main limitation of one-hot encoding compared to word embeddings?**
One-hot encoding does not capture semantic relationships between words and results in sparse, high-dimensional vectors. Word embeddings, on the other hand, are dense and capture contextual meaning.

---

### **18. What does the output of a CBOW model represent?**
The output represents the predicted **probability distribution** of words in the vocabulary, where the word with the highest probability is the most likely target word given the context.

---

### **19. What are hyperparameters that affect the CBOW model’s performance?**
- Window size  
- Embedding dimension  
- Learning rate  
- Number of epochs  
- Optimizer type (e.g., Adam, SGD)

---

### **20. How can we find similar words using the trained CBOW model?**
After training, we can use the **Gensim KeyedVectors** class and the `most_similar()` method to find words that have similar vector representations.

---

**Conclusion:**  
The Continuous Bag of Words (CBOW) model is a foundational NLP technique for learning word embeddings. It helps capture contextual relationships between words efficiently and is widely used in various natural language applications like text classification, translation, and chatbot systems.

---
