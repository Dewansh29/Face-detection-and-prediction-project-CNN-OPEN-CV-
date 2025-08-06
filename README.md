# Avengers Face Recognition using Deep Learning

![Avengers Banner](https://i.imgur.com/8xJg6b2.jpeg)

## 📖 Project Overview

This project is a high-accuracy, deep learning model built to recognize the faces of five main Avengers: Captain America (Chris Evans), Iron Man (Robert Downey Jr.), Thor (Chris Hemsworth), Black Widow (Scarlett Johansson), and The Hulk (Mark Ruffalo).

The journey to building this model was a challenging one, involving a multi-day process of intense debugging and problem-solving. Initial approaches hit a wall due to platform-specific bugs and the limitations of a small dataset. The final, successful approach pivoted to using **Transfer Learning with a pre-trained MobileNetV2 model**, which was then fine-tuned to achieve high accuracy and reliability.

This repository documents the final, successful workflow, which serves as a robust template for image classification tasks.

## ✨ Key Features

-   **High Accuracy:** Achieves over 98% validation accuracy by leveraging a state-of-the-art model.
-   **Transfer Learning:** Uses the powerful MobileNetV2 architecture, pre-trained on the ImageNet dataset.
-   **Advanced Training:** Employs robust training techniques, including learning rate scheduling (`ReduceLROnPlateau`) and early stopping (`EarlyStopping`) to achieve optimal results.
-   **Robust Data Pipeline:** Uses `tf.data.Dataset` for efficient, high-performance data loading and preprocessing, optimized for GPU environments.

## 📊 Dataset

The project uses the **Avengers Faces Classification** dataset, sourced from Kaggle. The dataset is pre-split into `train`, `test`, and `val` directories, with subdirectories for each of the five characters.

## 💻 Technology Stack & Libraries

This project was built using Python and the following key libraries:

-   **TensorFlow & Keras:** The core deep learning framework used for building, training, and fine-tuning the Convolutional Neural Network (CNN).
-   **OpenCV (`cv2`):** Used for loading, resizing, and color-space conversion of the images during the final testing phase.
-   **NumPy:** The fundamental package for numerical computation, used for handling image arrays.
-   **Matplotlib:** Used for visualizing the test images and their prediction results.
-   **OS & shutil:** Used for navigating the file system and preparing the data directories.

## ⚙️ Project Workflow: The Story of the Model

The core idea of this project is **Transfer Learning**. Instead of teaching a new model from scratch (like teaching a baby), we "hire an expert"—a powerful model that already understands images—and teach it a new, specific job.

---

### **Step 1: Prepare the "Briefing Documents" (Loading the Data)**

Before we can work with our expert model, we must organize our data efficiently.

-   We use **`tf.keras.utils.image_dataset_from_directory`**. This is a smart tool that looks at our `train` and `val` folders, reads all the images, and automatically labels them based on the folder they're in (e.g., all images in the `rdj` folder get the "rdj" label).
-   We add **`.cache()`** and **`.prefetch()`** to the dataset. This is like making photocopies of all our documents and laying them out on a table before a big meeting. It pre-loads the data into memory, making the training process on the GPU extremely fast.

---

### **Step 2: Assemble the "Team" (Building the Model)**

Here, we build our custom model by leveraging the pre-trained expert.

-   We load our expert, **`MobileNetV2`**, but we tell it **`include_top=False`**. This means we're only using its powerful visual analysis "brain," not its original decision-making part (which was trained to identify 1000 other things, like cats and dogs).
-   We **freeze** the expert's brain by setting **`base_model.trainable = False`**. This is the most important step. It protects the model's billions of parameters and years of experience from being accidentally damaged during our training.
-   We add our own new, small "manager" on top. These are our **`Dense`**, **`BatchNormalization`**, and **`Dropout`** layers. This is the only part that will be trained initially. Its only job is to look at the expert's complex analysis and make a final decision: "Which of the 5 Avengers is this?"

---

### **Step 3: The "Training Session" (The `model.fit` Process)**

This is where the actual learning happens.

-   The model is shown the training images over and over again (**`epochs`**). It makes a guess, checks the correct answer, and slowly adjusts its "manager" layers to get better.
-   We use smart tools (**callbacks**) to guide the learning process:
    -   **`ReduceLROnPlateau`**: If the model gets stuck and stops improving, this tool automatically reduces the learning rate, telling it to "take smaller, more careful steps."
    -   **`EarlyStopping`**: This tool is a supervisor. It watches the model's performance on the validation data. If the model stops improving for several epochs, it automatically stops the training to save time and prevent memorization (overfitting).
    -   **`ModelCheckpoint`**: This acts as our secretary, watching the entire session and saving only the absolute best version of the model it sees.

---

### **Step 4: The "Final Exam" (Testing)**

After training is complete, we must give our model a fair and honest final exam.

-   We load our best saved model from the training session.
-   We give it the **`test`** dataset—images it has never seen before in any context.
-   We must preprocess these test images in the **exact same way** we preprocessed the training images. This is crucial for getting accurate results.
-   The model makes its predictions, and we compare them to the true labels to see how well it *really* learned to generalize. The color-coding (green for correct, red for wrong) gives us an instant visual report card of its performance.
