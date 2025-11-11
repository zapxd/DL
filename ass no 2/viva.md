# Viva Questions and Answers  
## Assignment No. 2  
### Title: Implementing Feed Forward Neural Network with Keras and TensorFlow  

---

### **1. What is a Feed Forward Neural Network (FNN)?**
A Feed Forward Neural Network (FNN) is a type of artificial neural network where the connections between the nodes do not form a cycle. The information moves in one direction — from the input layer, through the hidden layers, to the output layer.

---

### **2. What is the MNIST dataset?**
MNIST (Modified National Institute of Standards and Technology) is a dataset of 70,000 handwritten digits (0–9) divided into 60,000 training images and 10,000 test images. Each image is 28x28 pixels in grayscale. It is commonly used for training and testing image classification models.

---

### **3. What is the main goal of this experiment?**
The goal is to train a feed-forward neural network (using Keras and TensorFlow) to classify handwritten digits from the MNIST dataset and achieve more than **90% accuracy** while maintaining low loss.

---

### **4. What is TensorFlow?**
TensorFlow is an open-source machine learning and deep learning framework developed by Google. It allows developers to create computational graphs for building and training neural networks.

---

### **5. What is Keras?**
Keras is a high-level deep learning API that runs on top of TensorFlow. It simplifies the process of building, training, and evaluating neural networks by providing easy-to-use abstractions like Sequential models and layers.

---

### **6. What is a Tensor in TensorFlow?**
A tensor is a multidimensional array used to represent data in TensorFlow. It can be a scalar, vector, matrix, or n-dimensional array, and it forms the basic data structure for all TensorFlow computations.

---

### **7. What are the layers used in your model?**
The model uses:
1. **Flatten Layer:** Converts the 2D 28x28 input images into a 1D array (vector).
2. **Dense Layer (Hidden Layer):** A fully connected layer with 64 neurons and ReLU activation.
3. **Dense Layer (Output Layer):** A fully connected layer with 10 neurons (for digits 0–9) and softmax activation.

---

### **8. What is the function of the Flatten layer?**
The Flatten layer reshapes a multi-dimensional input tensor into a single vector without changing its data. It prepares the image data for processing by the dense (fully connected) layers.

---

### **9. What is the Dense layer?**
A Dense layer (also called a fully connected layer) connects each neuron in the previous layer to every neuron in the current layer. It computes a weighted sum of the inputs, adds a bias, and passes the result through an activation function.

---

### **10. What is the role of the activation function in neural networks?**
Activation functions introduce non-linearity into the network, allowing it to learn complex patterns.  
Common activation functions:
- **ReLU (Rectified Linear Unit):** Sets all negative values to zero, speeding up convergence.
- **Softmax:** Used in the output layer for multi-class classification; converts outputs to probabilities.

---

### **11. Which optimizer did you use and why?**
We used **Stochastic Gradient Descent (SGD)** as the optimizer. It updates the weights based on a subset (batch) of the data, which helps the model converge faster and handle large datasets efficiently.

---

### **12. What loss function did you use and why?**
We used **Sparse Categorical Crossentropy** because it’s suitable for multi-class classification problems where the output labels are integers rather than one-hot encoded vectors.

---

### **13. Explain the `model.compile()` step.**
In the `compile()` step, we define:
- **Optimizer:** Algorithm to minimize loss (SGD in this case).
- **Loss Function:** Measure of model error (Sparse Categorical Crossentropy).
- **Metrics:** Performance measure (Accuracy).

Example:
```python
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

---

### **14. Explain the `model.fit()` function.**
The `fit()` function trains the neural network on the training data for a specified number of epochs. It also returns a `history` object containing details of loss and accuracy after each epoch.

---

### **15. What is an epoch?**
An epoch refers to one complete pass of the entire training dataset through the neural network during training.

---

### **16. What is the purpose of the `model.evaluate()` method?**
The `evaluate()` method calculates the overall loss and accuracy of the trained model on the test dataset to measure its performance.

---

### **17. What is the purpose of the `model.predict()` method?**
The `predict()` method generates predictions from the trained model. It returns probability values for each class, and the highest value determines the predicted class.

---

### **18. What does the softmax function do in the output layer?**
The softmax function converts raw output values (logits) into probability distributions across the 10 output classes, ensuring that all probabilities sum to 1.

---

### **19. Why is ReLU used as an activation function?**
ReLU (Rectified Linear Unit) improves training speed and avoids the vanishing gradient problem by keeping positive values unchanged and setting negative values to zero.

---

### **20. What is the difference between training accuracy and validation accuracy?**
- **Training Accuracy:** How well the model performs on the training data.
- **Validation Accuracy:** How well the model performs on unseen data during training.  
If there’s a large difference, the model may be overfitting.

---

### **21. How can you improve model generalization?**
Some techniques include:
- Using **Dropout layers**.
- **Data augmentation** (rotating, scaling images).
- **Early stopping** during training.
- **Regularization** (L1/L2 penalties).
- **Batch normalization**.

---

### **22. What is overfitting?**
Overfitting occurs when the model performs well on the training data but poorly on unseen data. It means the model has memorized the training data instead of learning general patterns.

---

### **23. What is the purpose of plotting accuracy and loss?**
Plotting helps visualize how the model is learning over epochs. It shows whether the loss is decreasing and accuracy is improving, which indicates proper training.

Example:
```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

---

### **24. What is a confusion matrix?**
A confusion matrix is a table used to visualize model performance by showing correct and incorrect classifications for each class. It helps identify where the model is making errors.

---

### **25. What accuracy did your model achieve?**
The model achieved **>90% accuracy** on the MNIST dataset after 10 epochs, meeting the target for this experiment.

---

### **26. What happens when the learning rate is too high or too low?**
- **Too High:** Model may overshoot minima and fail to converge.
- **Too Low:** Model converges very slowly or may get stuck in local minima.

---

### **27. Why do we normalize input data?**
Normalization scales the pixel values (0–255) to a smaller range (0–1), improving training stability and convergence speed.

Example:
```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

---

### **28. What is the role of the history object in Keras?**
The history object stores training metrics such as loss and accuracy for each epoch. It is used for performance analysis and visualization.

---

### **29. What is the difference between SGD and Adam optimizer?**
- **SGD:** Updates weights based on gradient and learning rate; simple and stable.
- **Adam:** Adaptive learning rate optimization that combines RMSprop and momentum, often converges faster.

---

### **30. What are some real-world applications of Feed Forward Neural Networks?**
- Handwritten digit recognition (like MNIST)
- Credit card fraud detection
- Stock market prediction
- Medical image analysis
- Sentiment analysis

---

**Conclusion:**  
This experiment demonstrates how to build, train, and evaluate a Feed Forward Neural Network using Keras and TensorFlow. The model successfully classified handwritten digits from the MNIST dataset with high accuracy, and visualization confirmed good training behavior and generalization.

---
