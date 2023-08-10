# Speeding Up Model Searching with Distributed Training in TensorFlow

When dealing with large datasets and complex deep learning models, training time can become a significant bottleneck. To accelerate the model searching process and reduce training time, we can leverage distributed training using TensorFlow's `MultiWorkerMirroredStrategy`.

## What is Distributed Training?

Distributed training is a technique used to train machine learning models on multiple GPUs or machines simultaneously. It involves distributing the training data and model across different devices and performing computations in parallel. This parallel processing leads to a significant reduction in training time, especially for large datasets and complex models.

## How Distributed Training Works in TensorFlow

In the provided code, we use `MultiWorkerMirroredStrategy` as the distributed training strategy. This strategy is designed for multi-GPU and multi-worker training scenarios.

1. **Data Parallelism**: In data parallelism, the dataset is divided into multiple partitions, and each partition is sent to different GPUs or workers. Each GPU processes its partition independently and computes the gradients for the model parameters.

2. **Model Parallelism**: In model parallelism, different parts of the model are assigned to different GPUs or workers. Each GPU computes the forward pass for its part of the model, and then the gradients are aggregated and updated collectively.

3. **Communication**: During training, the gradients from each GPU or worker are communicated to the others, and the model parameters are updated accordingly. The communication overhead is handled by TensorFlow's `MultiWorkerMirroredStrategy`.

## Impact of Distributed Training on Model Searching

In the provided code, we have defined multiple model architectures to search for the best performing one. By utilizing `MultiWorkerMirroredStrategy`, we can train these models simultaneously across multiple GPUs or workers.

1. **Parallelization**: With distributed training, each model architecture is trained on a different GPU or worker in parallel. This means that while one model is being trained, other models are also making progress simultaneously, which significantly speeds up the overall model searching process.

2. **Resource Utilization**: Distributed training allows us to fully utilize all available GPUs or workers, making the best use of our computational resources. This leads to better hardware resource utilization and faster model training.

3. **Reduced Training Time**: As training is performed in parallel, the overall time required to search for the best model is significantly reduced. This is especially beneficial when dealing with large datasets and complex models, as training on a single GPU or worker would take much longer.

4. **Improved Model Selection**: Faster model searching allows us to explore a larger range of hyperparameters and model architectures. This increases the likelihood of finding the best model with optimal hyperparameters, leading to improved model performance.

In conclusion, distributed training using `MultiWorkerMirroredStrategy` in TensorFlow provides a powerful solution to speed up the model searching process. By distributing the training across multiple GPUs or workers, we can effectively reduce training time, optimize resource utilization, and ultimately achieve better-performing models in a shorter amount of time. As a result, this technique is particularly beneficial when conducting extensive hyperparameter searches and model comparisons for large-scale image classification tasks.

Data set could be found here 
https://www.kaggle.com/datasets/ninadaithal/imagesoasis

Code example:
```
import tensorflow as tf
import os
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D
## Working model version 0.1

# Define the number of workers
num_workers = 2

# Initialize the distributed strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Define the data directories
current_dir = os.getcwd()
# Define the path to the original data folder
original_data_dir = current_dir + '/data-source/dementia/Data/'

# Define the path to the new data folder with train and test directories
train_data_dir = current_dir + '/test-data-source/dementia/train'
validation_data_dir = current_dir + '/test-data-source/dementia/test'

# Define some constants for the training
BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * num_workers
IMG_HEIGHT, IMG_WIDTH = 56, 56

# Prepare the data for training
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    validation_data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# AUTOTUNE is used to automatically tune the dataset prefetching based on available resources.
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# List to store validation accuracies for each model
validation_accuracies = []

# Define the model inside the strategy scope
with strategy.scope():
    # Define a list of model architectures with names to search for the best model
    model_architectures = [
        {
            'name': 'Dense Model',
            'model': tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(4, activation='softmax')
            ])
        },
        {
            'name': 'VGG16 Pre-trained Model',
            'model': Sequential([
                        VGG16(include_top=False, weights='imagenet', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
                        GlobalAveragePooling2D(),
                        Dense(4, activation='softmax')
                    ])
        },
        {
            'name': 'ResNet50 Pre-trained Model',
            'model': Sequential([
                        ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
                        GlobalAveragePooling2D(),
                        Dense(4, activation='softmax')
                    ])
        },
        {
            'name': 'Convo CNN Model',
            'model': Sequential([
                        Conv2D(64 * 2, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
                        MaxPooling2D((2, 2)),
                        Conv2D(128 * 2, (3, 3), activation='relu'),
                        MaxPooling2D((2, 2)),
                        Conv2D(256 * 2, (3, 3), activation='relu'),
                        MaxPooling2D((2, 2)),
                        Flatten(),
                        Dense(256 * 2, activation='relu'),
                        Dense(4 * 2, activation='relu'),
                        Dense(4, activation='softmax')
                    ]),

        }
    ]
    for model_info in model_architectures:
        model_name = model_info['name']
        model = model_info['model']
        print(f"Training {model_name} with architecture: {model}")

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['accuracy'])
        # Define the number of epochs
        num_epochs = 10

        # Calculate the time taken for data preprocessing and training
        start_time = time.time()
        # Start distributed training
        model.fit(train_dataset, epochs=num_epochs, validation_data=validation_dataset)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Convert elapsed time to hours
        hours = elapsed_time / 3600
        print(f"Training completed in {hours:.2f} hours.")
           # Evaluate the model on the validation dataset
        _, eval_acc = model.evaluate(validation_dataset)
        validation_accuracies.append((model_name, eval_acc))
        print(f"{model_name} validation accuracy: {eval_acc}")

# Function to count the number of image files in a directory
def count_images(directory):
    num_images = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                num_images += 1
    return num_images


# Count the number of images in the train directory
num_train_images = count_images(train_data_dir)
print(f"Number of images in the train directory: {num_train_images}")


# Find the best model with the highest validation accuracy
best_model_info = max(validation_accuracies, key=lambda x: x[1])
best_model_name, best_model_val_acc = best_model_info

print(f"\nBest model architecture: {best_model_name}")
print(f"Best model validation accuracy: {best_model_val_acc}")
```

The code is running on a Docker container. Make sure you installed nvidia driver for your GPU
```
# Use the TensorFlow GPU base image
FROM tensorflow/tensorflow:latest-gpu as Builder

# Install Conda
# Install required packages for downloading Miniconda
# Install PyCUDA
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    cuda-toolkit-12-2 \
    openssh-client

FROM tensorflow/tensorflow:latest-gpu as PipInstaller

# Copy system-level packages from the builder image
COPY --from=builder /usr/local/cuda /usr/local/cuda


# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container's working directory
COPY . .
# COPY requirements.txt .
# COPY data-source/ .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Set the entry point to run main.py when the container starts
# ENTRYPOINT ["python", "-m", "src.distributed_training"]
# ENTRYPOINT [ "python", "-m", "src.single_machine_training" ]
ENTRYPOINT ["python", "-m", "src.rework_distributed_from_mnist"]
# ENTRYPOINT [ "ls", "-la", "/root/.ssh/" ]
```

Run the script on Master node:
`sudo docker stop worker-0 | true && sudo docker rm worker-0 | true && sudo docker rmi --force my_tensorflow_app && sudo docker build -t my_tensorflow_app . && sudo docker run --gpus all -p 2222:2222 -e TF_CONFIG='{"cluster": {"worker": ["192.168.1.100:2222", "192.168.1.101:2222"]}, "task": {"type": "worker", "index": 0}}' --name worker-0  my_tensorflow_app`

Run the script on Worker node:
`sudo docker stop worker-1 | true && sudo docker rm worker-1 | true && sudo docker rmi --force my_tensorflow_app && sudo docker build -t my_tensorflow_app . && sudo docker run --gpus all -p 2222:2222 -e TF_CONFIG='{"cluster": {"worker": ["192.168.1.100:2222", "192.168.1.101:2222"]}, "task": {"type": "worker", "index": 1}}' --name worker-1  my_tensorflow_app
`
!!! Make sure there's ssh key from Master node to all Worker node !!!

========================================================================================================================================================
# Object reconition using pre-trained model on Waymo open data set
![Object Detection GIF](./media/animation.gif)
Waymo has an open data set for cameras and lidar for independent researcher to experiment their algorithm. This article are meant to process a single segment of camera recorded. How to parse the Image and apply a pre-trained object recognition into the frame data. 

```
import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import os

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import matplotlib.pyplot as plt
import io
import pyarrow.parquet as pq

# make sure to download this model to the same directory as this file
detection_model = tf.saved_model.load("./ssd_mobilenet_v2_2")
def download_and_run_model(pil_image):
    # Load the image
    image_rgb = np.array(pil_image)

    # Prepare the image for object detection
    input_tensor = tf.convert_to_tensor(image_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Perform object detection
    detections = detection_model(input_tensor)

    return detections


output_dir = './object-detection-results' 
# Save the image to the output directory
os.makedirs(output_dir, exist_ok=True)

def visualize_objects(image, detections, image_name):
    image_rgb = np.array(image)

    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.uint32)

    plt.imshow(image_rgb)

    for i in range(len(detection_boxes)):
        if detection_scores[i] > 0.5:
            ymin, xmin, ymax, xmax = detection_boxes[i]
            h, w, _ = image_rgb.shape
            left = int(xmin * w)
            top = int(ymin * h)
            right = int(xmax * w)
            bottom = int(ymax * h)
            rect = plt.Rectangle((left, top), right - left, bottom - top,
                                 fill=False, edgecolor='green', linewidth=2)
            plt.gca().add_patch(rect)

   
    output_path = os.path.join(output_dir, f"{image_name}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def process_one_frame(image_bytes, frame_name):
        
    # Convert the image bytes into a PIL image
    pil_image = Image.open(io.BytesIO(image_bytes))
    detections = download_and_run_model(pil_image)
    print(f"Frame name: {frame_name}")
    visualize_objects(pil_image, detections, f"{frame_name}")

def process_one_file(FILENAME):
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frame_name = frame.context.name
        process_one_frame(frame, frame_name)


# Define the path to the Parquet file
parquet_file_path = './training_camera_image_10017090168044687777_6380_000_6400_000.parquet'

# Open the Parquet file
parquet_table = pq.read_table(parquet_file_path)

# Convert the Parquet table to a Pandas DataFrame
df = parquet_table.to_pandas()

# Print the values of each column in the first row
# Extract two columns by name
selected_columns = df[['key.segment_context_name', '[CameraImageComponent].image']]

for index, row in selected_columns.iterrows():
    process_one_frame(row[1], f"{index}_{row[0]}")
```
