import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


# --- 1. Define Constants ---
DATASET_PATH = 'UTKFace'  # Path to the extracted dataset
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 20 # Start with 10, you can increase for better accuracy

# --- 2. Load and Preprocess Image Paths ---
# The filenames are like: [age]_[gender]_[race]_[date].jpg
# gender: 0 = Male, 1 = Female

image_paths = []
ages = []
genders = []

print("Loading dataset...")
for filename in os.listdir(DATASET_PATH):
    if filename.endswith('.jpg'):
        try:
            parts = filename.split('_')
            age = int(parts[0])
            gender = int(parts[1])
            
            # Filter out some bad labels
            if age > 0 and age < 117:
                image_paths.append(os.path.join(DATASET_PATH, filename))
                ages.append(age)
                genders.append(gender)
        except Exception as e:
            # Skip files with bad naming
            # print(f"Skipping {filename}: {e}")
            pass

if not image_paths:
    raise FileNotFoundError(f"No images found in {DATASET_PATH}. "
                           "Did you download and extract the dataset correctly?")

print(f"Total images loaded: {len(image_paths)}")

# Create a DataFrame
df = pd.DataFrame({'image_path': image_paths, 'age': ages, 'gender': genders})

# Split into training and validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# --- 3. Create tf.data Pipelines ---

def load_and_preprocess(image_path, age, gender):
    # Read image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Resize and preprocess (matches MobileNetV2 input)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    
    # Format labels
    age_label = tf.cast(age, tf.float32)
    gender_label = tf.cast(gender, tf.float32)
    
    # Our model will have two outputs, so we return a dictionary of labels
    return image, {'age_output': age_label, 'gender_output': gender_label}

def create_dataset(df, batch_size=BATCH_SIZE, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(
        (df['image_path'].values, df['age'].values, df['gender'].values)
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_ds = create_dataset(train_df, shuffle=True)
# Create a non-shuffled validation set for consistent evaluation
val_ds = create_dataset(val_df, shuffle=False) 

# --- 4. Build the "Multi-Output" Model ---
print("Building model...")

# Load MobileNetV2 base, pre-trained on ImageNet
base_model = MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,  # Don't include the final 1000-class layer
    weights='imagenet'
)

# Freeze the base model layers
base_model.trainable = False

# Define our model inputs
inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# We will fine-tune from the base model
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) # Add dropout for regularization

# --- Head 1: Age (Regression) ---
age_head = Dense(128, activation='relu')(x)
age_head = Dense(64, activation='relu')(age_head)
age_output = Dense(1, activation='linear', name='age_output')(age_head) # 'linear' for regression

# --- Head 2: Gender (Binary Classification) ---
gender_head = Dense(128, activation='relu')(x)
gender_head = Dense(64, activation='relu')(gender_head)
gender_output = Dense(1, activation='sigmoid', name='gender_output')(gender_head) # 'sigmoid' for binary

# Combine into a single model
model = Model(inputs=inputs, outputs=[age_output, gender_output])

# --- 5. Compile the Model ---
# We need separate loss functions for each head
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'age_output': 'mean_absolute_error',      # MAE is good for age regression
        'gender_output': 'binary_crossentropy'   # Standard for binary classification
    },
    metrics={
        'age_output': 'mae',
        'gender_output': 'accuracy'
    }
)

model.summary()


# --- 6. Train the Model ---
print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# --- 7. NEW: Plot Accuracy and Loss Curves ---
print("Plotting training history...")
history_dict = history.history

# Create a figure with 2x2 subplots
plt.figure(figsize=(14, 10))

# 1. Plot Age MAE (Loss)
plt.subplot(2, 2, 1)
plt.plot(history_dict['age_output_mae'], label='Train Age MAE')
plt.plot(history_dict['val_age_output_mae'], label='Val Age MAE')
plt.title('Age MAE (Regression Loss)')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

# 2. Plot Gender Accuracy
plt.subplot(2, 2, 2)
plt.plot(history_dict['gender_output_accuracy'], label='Train Gender Accuracy')
plt.plot(history_dict['val_gender_output_accuracy'], label='Val Gender Accuracy')
plt.title('Gender Accuracy (Classification)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 3. Plot Gender Loss
plt.subplot(2, 2, 3)
plt.plot(history_dict['gender_output_loss'], label='Train Gender Loss')
plt.plot(history_dict['val_gender_output_loss'], label='Val Gender Loss')
plt.title('Gender Loss (Binary Crossentropy)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 4. Plot Total Loss
plt.subplot(2, 2, 4)
plt.plot(history_dict['loss'], label='Total Train Loss')
plt.plot(history_dict['val_loss'], label='Total Val Loss')
plt.title('Total Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# --- 8. NEW: Classification Report and Confusion Matrix (for Gender) ---
print("Generating classification report and confusion matrix for Gender...")

# Get true labels from the validation dataset
y_true_gender = []
for images, labels in val_ds:
    y_true_gender.extend(labels['gender_output'].numpy())

y_true_gender = np.array(y_true_gender)

# Get predictions from the model on the validation set
# model.predict() returns a list: [age_preds, gender_preds]
predictions = model.predict(val_ds)
y_pred_gender_probs = predictions[1].squeeze() # Get gender preds and remove extra dim

# Convert probabilities to binary classes (0 or 1)
y_pred_gender = (y_pred_gender_probs > 0.5).astype(int)

# --- Classification Report ---
print("\nGender Classification Report:")
target_names = ['Male (0)', 'Female (1)']
print(classification_report(y_true_gender, y_pred_gender, target_names=target_names))

# --- Confusion Matrix ---
print("\nGender Confusion Matrix:")
cm = confusion_matrix(y_true_gender, y_pred_gender)
print(cm)

# Plot Confusion Matrix using Seaborn

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Gender Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# --- 9. Save Your Custom Model ---
model.save('age_gender_model.h5')
print("Model saved as age_gender_model.h5")