# YOLOv8 to TensorRT on Jetson Tutorial

## Project Overview

This tutorial aims to provide a step-by-step guide on how to convert a YOLOv8 detection model (segmentation models are similar) to TensorRT format and run it on Jetson devices. TensorRT is a deep learning inference library that optimizes models for high-performance inference. By converting the YOLOv8 model to TensorRT format, we can significantly improve the inference speed and efficiency on Jetson devices. This tutorial will cover all steps from environment setup, model conversion, to actual deployment and testing, ensuring that readers can successfully complete the model conversion and deployment process.

## Devices and Operating Systems

### PC
- **Operating System**: Ubuntu 22.04.4 LTS (Jammy Jellyfish)
- **CUDA Version**: 12.3

### Jetson Devices
1. **Jetson AGX Orin Developer Kit**
   - **JetPack SDK**: 5.1.1
   - **Operating System**: Ubuntu 20.04 (Focal)
   - **CUDA Version**: 11.4.315
   - **cuDNN Version**: 8.6.0.166
   - **TensorRT Version**: 8.5.2.2

2. **Jetson Xavier NX Developer Kit**
   - **JetPack SDK**: 5.1.1
   - **Operating System**: Ubuntu 20.04 (Focal)
   - **CUDA Version**: 11.4.315
   - **cuDNN Version**: 8.6.0.166
   - **TensorRT Version**: 8.5.2.2

## Environment Setup

### PC Setup

To prepare your PC environment, it is crucial to have Docker and the NVIDIA Container Toolkit installed, as we will be conducting our operations primarily within Docker containers. Follow the steps below to set up your PC:

1. **Install Docker Engine**

2. **Install NVIDIA Container Toolkit**

3. **Pull the Ultralytics Docker Image**:
    - Find the appropriate Docker image for your system and CUDA version on [Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics):
      ```sh
      docker pull ultralytics/ultralytics:8.2.32
      ```

### Jetson Setup

For the Jetson devices, since JetPack is already installed, you only need to install a few additional packages. These steps will be covered in the specific sections where they are required. Make sure your Jetson device is updated and ready.

## Converting YOLOv8 Model to ONNX Format

### Step-by-Step Guide on PC

1. **Prepare a Folder**:
    - Create a directory on your computer that will be mounted to the Docker container. For example:
      ```sh
      mkdir -p /home/chentong/Documents/DeepLearning/Data
      ```

2. **Clone the Repository**:
    - Clone the repository that provides scripts and instructions for converting YOLOv8 models to TensorRT:
      ```sh
      git clone https://github.com/triple-Mu/YOLOv8-TensorRT /home/chentong/Documents/DeepLearning/Data
      ```

3. **Run the Ultralytics Docker Image**:
    - Open the Ultralytics Docker image and mount the previously created folder:
      ```sh
      docker run -it --rm --ipc=host --gpus all -v /home/chentong/Documents/DeepLearning/Data:/opt/Data -w /opt/Data ultralytics/ultralytics:8.2.32
      ```

4. **Copy Your YOLOv8 Model**:
    - Copy your YOLOv8 model into the `YOLOv8-TensorRT` directory. Here, we use the official pretrained `yolov8s.pt` model as an example.

5. **Export YOLOv8 Model to ONNX**:
    - Follow the repository steps to export the model to ONNX format. Use the following command:
      ```sh
      python3 export-det.py \
      --weights yolov8s.pt \
      --iou-thres 0.65 \
      --conf-thres 0.25 \
      --topk 100 \
      --opset 11 \
      --sim \
      --input-shape 1 3 640 640 \
      --device cuda:0
      ```
    - Adjust the parameters as needed for your specific requirements.
    - After the export process completes, you should find the ONNX model in the same directory, named according to your original model file, e.g., `yolov8s.onnx`.

At this point, the work on the PC side is complete, and you should have an ONNX model ready for conversion to TensorRT on the Jetson device.

## Converting ONNX Model to TensorRT Format on Jetson

### Step-by-Step Guide on Jetson

1. **Clone the Repository**:
    - Clone the repository on your Jetson device:
      ```sh
      git clone https://github.com/triple-Mu/YOLOv8-TensorRT
      ```

2. **Copy ONNX Model and Test Images**:
    - Copy the ONNX model generated on your PC to the `YOLOv8-TensorRT` directory on your Jetson device.
    - Prepare a folder with test images, for example named `test_images`, and place it in the `YOLOv8-TensorRT` directory.

3. **Convert ONNX Model to TensorRT Format**:
    - Use the following command to convert the ONNX model to TensorRT format [(reference)](https://github.com/triple-Mu/YOLOv8-TensorRT/blob/main/docs/Jetson.md):
      ```sh
      /usr/src/tensorrt/bin/trtexec \
      --onnx=yolov8s.onnx \
      --saveEngine=yolov8s.engine \
      --fp16
      ```
    - Note that converting the model on Jetson might take a considerable amount of time.

After completing these steps, you will have a TensorRT engine file (`yolov8s.engine`) ready for deployment and inference on the Jetson device.

## Running the TensorRT Model on Jetson

### Step-by-Step Guide on Jetson

1. **Install Required Python Packages**:
    - Since the inference will use TensorRT, you need to install `cuda-python` or `pycuda`. Here are the installation commands:
      - Install `cuda-python` (make sure to match the version with your CUDA installation; in this case, CUDA 11.4, so we use 11.5.0 version of `cuda-python`):
        ```sh
        pip3 install cuda-python==11.5.0
        ```
      - Install `pycuda`:
        ```sh
        pip install pycuda
        ```

2. **Prepare for Inference**:
    - The repository provides a script `infer-det-without-torch.py` for inference without using PyTorch.
    - Ensure your TensorRT engine file (`yolov8s.engine`) and test images are in the same directory as `infer-det-without-torch.py`.

3. **Fixing np.bool Error**:
    - If you encounter an `np.bool` error, this is due to the deprecation of `np.bool` in numpy version 1.20 and later. TensorRT versions like 8.5.2.2 may call older numpy versions, causing this issue.
    - To fix this, locate the TensorRT code that includes `np.bool` and replace it with `bool`. For example:
      ```sh
      sudo nano /usr/lib/python3.8/dist-packages/tensorrt/__init__.py
      ```
    - Find and replace:
      ```python
      bool: np.bool,
      ```
      with:
      ```python
      bool: bool,
      ```
4. **Handling Import Errors with PyTorch**:
    - If PyTorch is not installed on the Jetson device, you need to comment out all contents of the `__init__.py` file in the `models` folder to prevent import errors:
      ```sh
      sudo nano /path/to/YOLOv8-TensorRT/models/__init__.py
      ```
    - Comment out all lines in the file by adding `#` at the beginning of each line.
    - 
5. **Run the Inference Script**:
    - Execute the inference script with the necessary parameters. You can also modify the script to calculate the time taken for each step:
      ```sh
      python3 infer-det-without-torch.py
      ```
    - Check the script for detailed parameter usage and adjust as needed.

### Sample Script Execution

Here's an example command to run the inference script. Adjust the parameters according to your setup:

```sh
python3 infer-det-without-torch.py --engine yolov8s.engine --input_folder test_images --output_folder output_images
```

You might need to tweak the script or the parameters to suit your specific needs, such as calculating the time taken for each inference step.

### References
- [YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT)
- [ultralytics](https://github.com/ultralytics/ultralytics)
