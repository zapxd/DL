# Viva Questions and Answers  
## Assignment No. 3  
### Title: Image Classification using Convolutional Neural Network (CNN)  

---

### **1. What is Image Classification?**
Image classification is the process of categorizing and labeling images into specific classes based on their visual features using machine learning or deep learning techniques.

---

### **2. What is a Convolutional Neural Network (CNN)?**
A CNN is a type of deep learning model designed for processing image data. It automatically detects spatial patterns like edges, textures, and shapes through convolutional and pooling operations.

---

### **3. Why are CNNs better than traditional neural networks (MLPs) for image tasks?**
CNNs automatically extract spatial features from images using local connections and shared weights, while traditional neural networks treat all pixels independently and require manual feature extraction.

---

### **4. What are the main layers of a CNN?**
1. **Convolutional Layer** – Extracts spatial features.  
2. **ReLU Activation Layer** – Introduces non-linearity.  
3. **Pooling Layer** – Reduces feature dimensions.  
4. **Flatten Layer** – Converts 2D data to 1D.  
5. **Dense (Fully Connected) Layer** – Performs classification.  
6. **Softmax Output Layer** – Produces probabilities for each class.

---

### **5. What is the role of the Convolutional layer?**
It applies filters (kernels) over the image to extract local features such as edges, textures, and corners, forming feature maps that capture spatial relationships.

---

### **6. What is the purpose of the Pooling layer?**
Pooling reduces the size of feature maps by keeping only the most important information. It helps control overfitting and reduces computation.

---

### **7. What is the function of the ReLU activation?**
ReLU (Rectified Linear Unit) replaces negative pixel values with zero, introducing non-linearity and improving training speed and convergence.

---

### **8. Why is the Softmax function used in the output layer?**
Softmax converts the model’s outputs into probabilities that sum to 1, helping determine which class the image most likely belongs to.

---

### **9. What is the significance of Flattening in CNNs?**
Flattening converts 2D feature maps into a 1D vector, preparing the data for processing by fully connected (Dense) layers.

---

### **10. What is overfitting and how can it be prevented?**
Overfitting happens when the model performs well on training data but poorly on new data. It can be reduced using dropout, data augmentation, early stopping, and regularization.

---

### **11. What is model generalization?**
Model generalization refers to a model’s ability to perform well on unseen data, not just on the data it was trained on.

---

### **12. What is the purpose of normalization in image preprocessing?**
Normalization scales pixel values to a smaller range (e.g., 0–1), improving model stability, convergence speed, and accuracy.

---

### **13. What is the optimizer’s role in training CNNs?**
An optimizer updates the model’s weights during training to minimize the loss function and improve prediction accuracy.

---

### **14. What is the difference between training accuracy and test accuracy?**
- **Training Accuracy:** Accuracy on data used for training.  
- **Test Accuracy:** Accuracy on unseen data used to evaluate generalization.

---

### **15. What does high accuracy and low loss indicate about the model?**
It means the model has learned well, makes confident predictions, and generalizes effectively to unseen data.

---

### **16. What was the accuracy achieved by your CNN model?**
The CNN achieved around **99% accuracy** on the MNIST dataset, indicating strong performance and generalization.

---

### **17. What are some real-world applications of CNNs?**
- Face and object recognition  
- Medical image analysis  
- Self-driving cars  
- Security and surveillance  
- Optical character recognition (OCR)

---

**Conclusion:**  
Convolutional Neural Networks (CNNs) are highly effective for image classification tasks. They automatically learn spatial hierarchies of features, require minimal preprocessing, and achieve excellent accuracy, making them a foundational tool in modern computer vision.

---
