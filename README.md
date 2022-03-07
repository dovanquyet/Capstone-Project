# Workers Supervision for Construction Safety
## 1. Preliminary Report
- [x] Introduction: (200 words)
- [x] Dataset: (200 words)
- [x] Methodology: (400 words)
- [x] Preliminary Results (300-350 words)
## 2. Project Progress
### 2.1. Colleting and Preprocessing data
- [x] Multi-Person Detector
  - [x] HABBOF
  - [x] CEPDOF
- [x] Binary Image Classifier
  - [x] Safety Helmet: https://drive.google.com/file/d/18sbXg2GpcKz_H1URr-6M26VStj0L1Enj/view?usp=sharing
  - [X] Safety Jacket: https://drive.google.com/file/d/1hFaayAS9cP4xEmp4GbF9sVMz7PSBZYGF/view?usp=sharing
### 2.2. Architecture Implementation
- [x] Multi-Person Detector
- [x] Binary Image Classifier
### 2.3. Training Process
- [x] Multi-Person Detector
- [X] Binary Image Classifier 
- [x] Fisheye augmentation (novel)
### 2.4. Inference and Evaluation
- [X] Multi-Person Detector  
- [X] Binary Image Classifier 
## 3. Project Layout
```
project
├── config
|   ├── config.yaml: Configuration file for training and inference processes
├── core
|   ├── augmentations
|   |   ├── detection.py: Customized augmentation methods for preprocessing datasets used in training the detector  
|   ├── criterions
|   |   ├── angle_losses.py: Periodic L1 and L2 losses from angle prediction of our detector
|   |   ├── rotation_aware_loss.py: Total loss of our detector for detecting bounding boxes' center, size and angle of rotation
|   ├── datasets
|   |   ├── custom.py: Customized dataset reader used to pass dataset into training process of the detector 
|   |   ├── fisheye_effect.py: Apply fisheye distortion to images crawled from Internet which may have dissimilar features comparing to fisheye images
|   ├── logging
|   |   ├── logger.py: Logger used to display training progress
|   |   ├── training_monitor.py: Drawing and saving loss during training process
|   ├── models
|   |   ├── classification
|   |   |   ├── efficientnet.py: EfficientNets general architecture implementation
|   |   ├── detection
|   |   |   ├── backbones.py: Implementation of RAPiD's components
|   |   |   ├── rapid.py: RAPiD's full architecture assembling
|   ├── utils
|   |   ├── detection.py: Helper functions to calculate Intersection of Unions (IoUs) between bounding boxes
├── documents
|   ├── proposal.pdf: Brief information about project
|   ├── presentation.pptx: Slides for project presentation
|   ├── Math4995_MilestoneReport_poster.pptx: Poster report for preliminary results
├── inputs: Directory contains all inputs for inference including videos (*.mp4) and images
├── outputs: Directory contains all according inference outputs
├── scripts
|   ├── inference.py: Inference code of the whole project
└── └── train.py: Training code for both detector and classifier
