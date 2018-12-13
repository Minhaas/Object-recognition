# Object-recognition
Simple project on object recognition in self driving cars using python



Step 1: Create an Anaconda environment with python version 3.6.

conda create -n retinanet python=3.6 anaconda


Step 2: Activate the environment and install the necessary packages.

source activate retinanet
conda install tensorflow numpy scipy opencv pillow matplotlib h5py keras


Step 3: Then install the ImageAI library.

pip install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.1/imageai-2.0.1-py3-none-any.whl

Step 4: Now download the pretrained model required to generate predictions. This model is based on RetinaNet. Click this link to download the same. https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5


Step 5: Copy the downloaded file to your current working folder

Step 6: Download the image. Name the image as image.png

Step 7: Open jupyter notebook (type jupyter notebook in your terminal) and run the codes.

This will create a modified image file named image_new.png, which contains the bounding box for your image.

Step 8: To print the image use the file print.py





