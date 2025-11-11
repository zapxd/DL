# Viva Questions and Answers  
## Assignment No. 4  
### Title: Anomaly Detection using Autoencoder  

---

### **1. What is an Autoencoder?**
An Autoencoder is an unsupervised neural network that learns to reconstruct its input. It compresses data into a lower-dimensional **latent representation (encoding)** and then reconstructs it back to the original input using a **decoder**.

---

### **2. What is the structure of an Autoencoder?**
An Autoencoder consists of three main parts:
1. **Encoder:** Compresses input data into a lower-dimensional latent space.  
2. **Latent Space:** Represents compressed features (bottleneck layer).  
3. **Decoder:** Reconstructs the original input from the latent representation.

---

### **3. What is the main purpose of using Autoencoders in anomaly detection?**
Autoencoders are used to learn the patterns of normal data. During inference, anomalies (unusual data) show higher reconstruction errors since they deviate from the learned normal patterns.

---

### **4. How does anomaly detection using Autoencoders work?**
1. Train the Autoencoder only on **normal data**.  
2. Use the trained model to reconstruct new data.  
3. Calculate **reconstruction error** (difference between input and output).  
4. Data points with reconstruction error above a certain **threshold** are considered anomalies.

---

### **5. What type of learning does an Autoencoder perform?**
Autoencoders perform **unsupervised learning** because they do not require labeled data — the network learns to map input to itself.

---

### **6. What is meant by "latent representation"?**
A latent representation is a compressed form of the input data created by the encoder. It captures the essential features and structure of the data in fewer dimensions.

---

### **7. Why is Autoencoder suitable for anomaly detection?**
Because it learns to reconstruct **normal patterns**, data that doesn’t conform (anomalies) produces high reconstruction errors, making it easy to detect unusual behavior.

---

### **8. What is reconstruction error?**
Reconstruction error is the difference between the original input and the reconstructed output. It quantifies how well the Autoencoder has learned to reproduce input data.

---

### **9. How do you determine anomalies using reconstruction error?**
By setting a **threshold value**.  
- If the reconstruction error > threshold → **Anomaly**  
- If the reconstruction error ≤ threshold → **Normal**

---

### **10. What dataset was used in this experiment?**
An **ECG dataset** was used, where label **1** indicates normal observations and label **0** indicates anomalies.

---

### **11. Why do we train Autoencoders only on normal data?**
Because the model needs to learn the pattern of normal behavior. When it encounters anomalies, it will fail to reconstruct them accurately, resulting in a higher reconstruction error.

---

### **12. What is the role of normalization or scaling in this model?**
Normalization (such as using StandardScaler) helps stabilize and speed up training by ensuring all features are on a similar scale, which improves convergence and performance.

---

### **13. What optimizer and loss function are commonly used in Autoencoders?**
- **Optimizer:** Adam or SGD (for adjusting weights during training).  
- **Loss Function:** Mean Squared Error (MSE), which measures the difference between input and reconstructed output.

---

### **14. What does a high reconstruction error indicate?**
A high reconstruction error indicates that the data point does not follow the learned normal pattern — it is likely an **anomaly**.

---

### **15. What is the advantage of using Autoencoders for anomaly detection?**
- Effective for **imbalanced datasets** where normal data is abundant but anomalies are rare.  
- Automatically learns complex feature representations without manual feature engineering.  
- Can handle **high-dimensional data** efficiently.

---

### **16. What are the main limitations of Autoencoders?**
- They require large amounts of normal data to learn effectively.  
- Sensitive to hyperparameters and architecture design.  
- May not generalize well if normal data is too diverse.  

---

### **17. How do you choose a threshold for anomaly detection?**
By analyzing the distribution of reconstruction errors on normal data.  
A threshold can be set using methods like:
- Statistical methods (e.g., mean + 3×standard deviation)  
- Visual inspection of error plots  

---

### **18. What is meant by an imbalanced dataset?**
An imbalanced dataset is one where one class (e.g., normal data) heavily outnumbers the other (e.g., anomalies), making traditional classification methods ineffective.

---

### **19. How can you improve the precision and recall of the model?**
- Use a more complex architecture.  
- Tune hyperparameters (learning rate, number of layers, neurons).  
- Add more relevant features.  
- Experiment with different threshold values or algorithms.

---

### **20. What evaluation metrics can be used for anomaly detection?**
- Precision  
- Recall  
- F1-score  
- ROC-AUC (Receiver Operating Characteristic Curve - Area Under Curve)

---

### **21. Why is high accuracy not always meaningful in anomaly detection?**
In imbalanced datasets, the majority class dominates. A model can achieve high accuracy by predicting all samples as normal but fail to detect anomalies (low recall).

---

### **22. What is the difference between Encoder and Decoder?**

| Aspect | Encoder | Decoder |
|---------|----------|----------|
| **Purpose** | Compresses input data | Reconstructs data from latent representation |
| **Output** | Latent space representation | Reconstructed input |
| **Direction** | Input → Latent | Latent → Output |

---

### **23. What happens if we train the Autoencoder with both normal and anomalous data?**
The model may learn to reconstruct anomalies as well, reducing the difference in reconstruction errors and making anomaly detection ineffective.

---

### **24. What does the bottleneck layer represent in an Autoencoder?**
It represents the most compact and informative representation of the input data — capturing essential patterns while discarding noise.

---

### **25. What are real-world applications of Autoencoders in anomaly detection?**
- Credit card fraud detection  
- Network intrusion detection  
- Industrial equipment fault detection  
- Medical anomaly detection (ECG, MRI)  
- Cybersecurity and fraud analytics  

---

**Conclusion:**  
Autoencoders are powerful tools for anomaly detection in imbalanced datasets. By learning to reconstruct normal data, they identify anomalies through high reconstruction errors. Their effectiveness lies in their ability to capture essential patterns in data and generalize well to unseen inputs.

---
