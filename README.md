# project1
1. ABSTRACT:-

This project presents a real-time facial emotion recognition system that enhances user experience by recommending music and movies based on the user’s current mood. Leveraging deep learning models, the system processes facial expressions captured through grayscale images and classifies them into one of seven emotional categories: happy, sad, angry, fear, disgust, surprise, or neutral. These detected emotions then guide personalized media recommendations—such as upbeat playlists for happy users or calming music for stressed individuals.

To develop this system, we utilized the FER-2013 dataset, which contains over 32,000 48x48 grayscale facial images. The model is built using TensorFlow and Keras frameworks, incorporating efficient and scalable architectures like MobileNetV2 and EfficientNetB0 through transfer learning. These pre-trained convolutional neural networks (CNNs) extract facial features and classify emotions with high accuracy, while techniques like early stopping ensure optimal training. Integrated APIs like Spotify provide curated playlists, and movie recommendations are filtered based on emotional alignment using metadata and genre profiling. The proposed system demonstrates promising accuracy and responsiveness, offering a seamless, emotion-aware entertainment experience.

2. INTRODUCTION:-

In today’s digital landscape, personalization is key to user satisfaction. Emotion-aware systems, which detect and respond to users' emotional states, are rapidly gaining popularity, especially in entertainment and user engagement domains. This project, Real-Time Mood Detection Music and Movie Recommendations, aims to detect facial emotions in real-time and recommend suitable music and movies to match the user’s mood. By integrating computer vision, deep learning, and content recommendation strategies, we enhance the user's interaction with multimedia platforms in a meaningful way.
The foundation of our system is a convolutional neural network trained on the FER-2013 dataset—a collection of 48x48 grayscale images labeled with one of seven emotional states. These small, preprocessed images make the dataset ideal for training deep learning models while maintaining high computational efficiency. To optimize performance, we use transfer learning with MobileNetV2 and EfficientNetB0, two lightweight yet powerful CNNs that offer fast inference without sacrificing accuracy. These models extract relevant features from facial images and classify emotions effectively, even in real-time environments.
Once an emotion is detected, the system provides targeted recommendations. A happy user might receive energetic music and uplifting movies, while someone feeling sad may be presented with comforting songs or emotional dramas. Spotify’s API is used for music suggestions, while movie recommendations are driven by mood-aligned metadata and sentiment-based filtering. This dual recommendation pipeline ensures the content is not only relevant but emotionally resonant.
Overall, our system illustrates the potential of AI in enhancing digital entertainment through emotional intelligence. It serves as a step toward more empathetic and intuitive user experiences by aligning technology with human emotions.
 

3. ARCHITECTURE:-

The architecture of the emotion recognition system is centered around a deep Convolutional Neural Network (CNN) framework, optimized through transfer learning using state-of-the-art pre-trained models such as MobileNetV2 and EfficientNetB0. These models are specifically chosen due to their proven balance between high accuracy and computational efficiency, making them ideal for real-time applications. The model takes as input a 48x48 grayscale image of a human face—sourced from the FER-2013 dataset or captured live via webcam—and processes it through a standardized preprocessing pipeline. This involves facial alignment, resizing, and normalization to ensure consistency in scale, position, and pixel distribution. These preprocessing steps significantly improve the model's ability to generalize across various lighting conditions, angles, and facial structures.

Once preprocessed, the image is passed into the base feature extractor—either MobileNetV2 or EfficientNetB0—where the convolutional layers are kept frozen to retain pre-learned filters that detect edges, textures, and facial components like eyes, mouth, and brows. These features are highly transferable and help in recognizing subtle facial expressions. The output from these frozen layers is then funneled through a Global Average Pooling layer, which compresses the spatial dimensions of the feature maps into a flat vector. This pooling technique not only reduces overfitting by minimizing the number of parameters but also improves computational efficiency by avoiding fully connected layers at this stage.

The pooled features are then passed through custom dense (fully connected) layers, which act as the classifier head of the model. These layers are trained on the emotion classification task and are accompanied by Dropout layers that randomly deactivate neurons during training to prevent overfitting and promote robustness. The final classification is performed by a Dense layer with a softmax activation function, which outputs a probability distribution over seven predefined emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The class with the highest probability is considered the predicted emotional state.

This architecture is trained using the Adam optimizer with a learning rate of 2e-4 and the categorical cross-entropy loss function, both of which are well-suited for multi-class classification problems. An EarlyStopping callback is employed to monitor validation loss and halt training once performance stagnates, thus preserving the model with the best generalization capability. The modularity of the architecture allows for easy experimentation with other pre-trained models or layer configurations, and its lightweight nature ensures fast inference times, making it ideal for real-time facial emotion recognition tasks in both mobile and web-based environments. This architectural setup forms the backbone of the entire system, enabling accurate emotion detection that feeds into a downstream recommendation engine for mood-specific music and movie suggestions.

 



 
 
4. Load and Preprocess Dataset

The dataset used for training and evaluation in this project is the FER-2013 dataset, which contains grayscale facial images categorized into seven emotion classes. These images are stored in a directory structure compatible with Keras’s flow_from_directory function. To prepare the data for training, the images are first rescaled to values between 0 and 1 by dividing the pixel values by 255. Additionally, data augmentation techniques are applied to the training data using ImageDataGenerator. These include random rotations, zooms, horizontal flips, and small shifts in height and width. This not only increases the diversity of the training set but also helps improve the model's ability to generalize to unseen data. The images are resized to 96x96 pixels and loaded in batches using the Keras data generator, which simplifies the feeding of data into the model during training and validation.



 

5. Handle Class Imbalance with Class Weights

The FER-2013 dataset has an uneven distribution of samples across different emotion classes. To address this issue, class weights are computed using Scikit-learn’s compute_class_weight function. This ensures that the model does not become biased toward the more frequent classes during training. The weights are calculated based on the inverse frequency of each class, and these weights are passed to the fit() function during training. This approach penalizes the model more for making errors on underrepresented classes, thus improving overall accuracy and fairness in prediction across all emotions.

6. Build Model Architecture (MobileNetV2 + EfficientNetB0 Ensemble)

The core of the model architecture is an ensemble that combines two powerful pre-trained convolutional neural networks: MobileNetV2 and EfficientNetB0. Both models are initialized with weights trained on the ImageNet dataset and are set to non-trainable (frozen) to retain their pre-learned features and avoid overfitting. Each network processes the same input image (a 96x96 RGB face image), extracting rich hierarchical features from the facial expressions. Their outputs are passed through a GlobalAveragePooling2D layer, which reduces the spatial dimensions and produces a flat feature vector. A Dropout layer follows to enhance generalization by randomly disabling some neurons during training. Finally, each branch has a Dense output layer with a softmax activation to produce class probabilities across the seven emotions. The ensemble is created by averaging the predictions of both models using the Average() layer. This results in a final model that combines the strengths of both architectures, yielding a robust and accurate emotion classifier. The model is compiled using the Adam optimizer with a learning rate of 0.0002 and categorical crossentropy as the loss function—appropriate for multi-class classification tasks.

7. Model Training with EarlyStopping

To train the model efficiently while preventing overfitting, an EarlyStopping callback is used. This monitors the validation accuracy and stops training if there is no improvement for three consecutive epochs. The training process uses the earlier computed class weights to handle class imbalance and is set to run for a maximum of 20 epochs. The model is trained using the fit() method, which takes the training and validation datasets, along with the defined callbacks and class weights. During training, the best-performing model based on validation accuracy is retained. After training, the final model is saved as an .h5 file, enabling future reuse for inference without retraining.

8. Visualizing Model Performance

To evaluate how well the model learns over time, training and validation metrics are plotted. This includes both accuracy and loss curves. Using Matplotlib, a dual-subplot figure is created—one for plotting training and validation accuracy across epochs and another for plotting the corresponding loss values. These visualizations help diagnose potential issues such as underfitting or overfitting. For instance, if the training accuracy keeps increasing while validation accuracy stagnates or drops, it may suggest overfitting. Conversely, a consistently low accuracy could indicate the need for architectural tuning or better data preprocessing.



 
9. Real-Time Emotion Detection Using Webcam

One of the most engaging aspects of this project is its real-time emotion detection capability, which is implemented using OpenCV and the Haar Cascade classifier for face detection. The system loads the trained ensemble model from the previously saved .h5 file and initializes a video stream using the device’s webcam. Each video frame is converted to grayscale and scanned for faces using the Haar Cascade algorithm. For every detected face, the region is cropped, resized to 96x96 pixels, normalized (pixel values scaled to 0–1), and then passed to the trained CNN ensemble for prediction.

The model outputs a probability vector for the seven emotions, from which the top three predicted emotions are identified and displayed directly on the video feed, along with their respective confidence scores. A bounding box is drawn around the face for clarity, and predictions are updated in real-time. Additionally, to enhance reliability, the system maintains a rolling buffer of predictions using a deque (double-ended queue). After the webcam session ends (e.g., by pressing 'q'), the most frequently predicted emotion across all frames is selected as the final detected mood.



 

 
10. Manual Mood Selection and Recommendation System

To complement the automated detection system, the project also offers a manual mood selection option for users who prefer to input their emotion directly. This is especially useful in cases where the user wants personalized recommendations but cannot or does not wish to use the webcam-based detection. The user is presented with a list of seven emotions (happy, sad, angry, fear, surprise, neutral, disgust) and selects one by entering the corresponding number. Once the mood is selected, the system fetches and displays movie and music recommendations relevant to the chosen emotion.

 
11. Movie Recommendation via TMDb API

The movie recommendation component is powered by The Movie Database (TMDb) API, which is queried based on the emotion-derived genre mapping. Each emotion is mapped to a specific movie genre using TMDb’s internal genre IDs (e.g., 'happy' to comedy, 'sad' to drama, 'angry' to action). A random page number is generated to ensure variety, and the API is queried for top-rated movies within that genre that have a substantial vote count. The API returns a list of movie titles along with their release years, which are formatted and presented to the user. This approach makes the movie suggestions both emotionally relevant and diverse.



 
12. Music Recommendation via Spotify API

While not detailed in the code cells provided here, the project is designed to also integrate with the Spotify API for music recommendation. Based on the detected or manually selected emotion, the system could fetch curated playlists—either predefined or dynamically retrieved from Spotify. This extends the emotional resonance of the recommendation system by aligning music tones and genres with the user's emotional state. For instance, "Angry" might return playlists like “pov: you’re pissed off at the world,” while "Sad" could return soothing acoustic or piano tracks.



 
13. Final Output and User Interaction

At the end of a session—either through real-time detection or manual input—the system outputs a clear summary: the identified emotion, suggested movies, and recommended music playlists. This interaction closes the loop between emotion recognition and content personalization, turning a deep learning model into a user-friendly experience. The result is a practical and intuitive tool that not only demonstrates machine learning competence but also provides real-world value in mental wellness and entertainment personalization.



 
 
