# Semantic-Segmentation-of-Satelite-Imagery
This Project is Semantic Segmentation of High-Resolution Multi-Spectral Optical Satellite Images: A Deep Learning-based Approach for Monitoring Deforestation

# Introduction
This project aims to detect deforestation using semantic segmentation techniques applied to satellite imagery. The project utilizes the U-Net model implemented in Keras to perform the segmentation task. After training the model for 5 epochs, an impressive validation accuracy of 98% was achieved.

# Dataset
The dataset used for this project consists of 20 satellite images in the .tif format, along with their corresponding masks in the .shp (shapefile) format. The images represent various areas of interest where deforestation may have occurred. The masks provide pixel-level annotations indicating deforested regions.

# Model Architecture
The U-Net model architecture is employed for the semantic segmentation task. This architecture is well-known for its effectiveness in image segmentation tasks. It consists of an encoder path and a decoder path, where the encoder captures the contextual information and the decoder reconstructs the segmentation map.

# Usage
1. Ensure that you have the necessary dependencies installed, including Keras, TensorFlow, and any additional libraries mentioned in the requirements file.
2. Clone the repository to your local machine or download it as a ZIP file.
3. Place the satellite images (.tif) in the 'train' directory and the corresponding masks (.shp) in the same directory.
4. Run the 'main.ipynb' file.

# License
This project is licensed under the MIT License.
