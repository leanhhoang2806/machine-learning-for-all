# Setting Up Distributed Training at home using TensorFlow and Docker


## Introduction
Distributed training with TensorFlow on a home ethernet network allows you to leverage the combined computational power of multiple machines, accelerating your machine learning models without relying on expensive cloud resources. This tutorial will guide you through the process of setting up distributed training using TensorFlow and Docker containers. By the end of this tutorial, you'll have a fully functional distributed training setup, maximizing the potential of your home network.


### Motivation: The Need for Distributed Training
As the field of machine learning continues to advance, the models we develop are becoming increasingly powerful and capable of solving complex real-world problems. However, this progress comes at the cost of larger and more intricate models. With this growing size and complexity, training these models on a single machine can be a daunting task, often exceeding the memory and processing capabilities of even high-end hardware.


The motivation for distributed training lies in addressing the challenges posed by these large models and datasets. Here are some key reasons why distributed training is essential when dealing with massive models:


1. **Memory Constraints**: Large models require substantial memory that could not be stored on a single machine. LLM model has billions of parameters


2. **Faster Training**: More computer => more computing power => faster training


3. **Handling Big Data**: Distributed training leverages data parallelism, where each machine processes a subset of the data simultaneously.
4. **Scalability**: To have more computing power, just add more computer and ethernet cable. 
## Requirements
1. Multiple Computers: At least two computers with NVIDIA GPU on each computer (NVIDIA allowed CUDA to be utilized, AMD GPU can't work on this tutorial).
2. Ethernet Network: Need an Ethernet hub, that connect to wifi then connect to the two computers (for faster data tranferring).
3. All computers running `Ubuntu 22.04`
![Alt Text](/media/IMG_4958.HEIC)



## Step 1: Preparing Your Computers
Ensure that all computers involved meet the minimum system requirements for running TensorFlow. Install Docker on each machine following the official instructions from the Docker website.


## Step 2: Setting Up the Ethernet Network
1. Connect all the computers to the same ethernet network switch/router using ethernet cables. Ensure stable connections.


## Step 3: Figure out local network IPs
1. In the computer command line type
`ifconfig`
the IP address will look like:
```inet 192.168.1.17 netmask 0xffffff00 broadcast 192.168.1.255```


## Step 3: Configuring SSH Access
1. Install SSH on all computers if not already present.
2. Generate SSH keys on each machine and exchange them to enable passwordless access between machines.
3. Test SSH connectivity between machines to ensure seamless communication.


## Step 4: Creating a TensorFlow Docker Image
1. Create a Dockerfile that specifies the TensorFlow version and any additional dependencies required for your project.
2. Build the Docker image on each machine using the Dockerfile you created. This ensures consistency across all machines.


## Step 5: Configuring TensorFlow for Distributed Training
1. Update your TensorFlow code to enable distributed training. TensorFlow provides the tf.distribute module to easily set up distributed training.
2. Implement data parallelism or model parallelism strategies in your TensorFlow code, depending on your specific model and dataset.


## Step 6: Sharing Data with Docker Containers
1. Ensure that your dataset is accessible from all machines by either sharing it through network-attached storage (NAS) or synchronizing it across machines.
2. Create a shared folder or mount network storage to your Docker containers, allowing them to access the dataset during training.


## Step 7: Launching Distributed Training with Docker
1. Start a Docker container on each machine using the TensorFlow Docker image you created earlier.
2. Inside each container, run your distributed TensorFlow script, specifying the role of the container (worker, parameter server, etc.) and the IP addresses of other machines.


## Step 8: Monitoring and Troubleshooting
1. Monitor the training process using TensorBoard or other monitoring tools to keep track of metrics and identify potential issues.
2. If you encounter errors, inspect log files inside the Docker containers and double-check network configurations to ensure seamless communication between machines.


## Conclusion
Congratulations! You have successfully set up distributed training on your home ethernet network using TensorFlow and Docker containers. By harnessing the combined computational power of multiple machines, you can now tackle larger and more complex machine learning tasks efficiently. This setup not only saves costs but also empowers you to explore cutting-edge research and create state-of-the-art machine learning models from the comfort of your home. Happy training!





