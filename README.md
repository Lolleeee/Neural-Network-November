# Neural-Network-November
Homework 1 and 2 of Artificial Neural Network and Deep Learning course.

## Homework 1

### Overall description
In this assignment, we will classify 96x96 RGB images of blood cells. 
These images are categorized into eight classes, each representing a particular cell state. 
This is a multi-class classification problem, so our goal is to assign the correct class label to each RGB image.

### Data:
This dataset consists of images designed for the classification of different types of blood cells. 
Each image is labeled with one of eight classes, representing various blood cell types such as basophils, neutrophils, and lymphocytes.

**Dataset details**
- Image Size: 96x96
- Color Space: RGB (3 channels)
- Input Shape: (96, 96, 3)
- File Format: npz (Numpy archive)
- Number of Classes: 8

**Class Labels**
- 0: Basophil
- 1: Eosinophil
- 2: Erythroblast
- 3: Immature granulocytes
- 4: Lymphocyte
- 5: Monocyte
- 6: Neutrophil
- 7: Platelet

**Dataset Structure**
The training data is provided in the train_data.npz file, which contains:
- images: A Numpy array of shape (13759, 96, 96, 3) containing the RGB images.
- labels: A Numpy array of shape (13759,), with class values ranging from 0 to 7, corresponding to the classes listed above.

## Terms and Conditions
Participants in this competition must adhere to the following rules:
- Compliance: All participants must ensure their submissions comply with the guidelines provided throughout the competition. Any violation may result in disqualification.
- Originality: Submitted work must be original. Any form of plagiarism or unauthorized collaboration is strictly prohibited.
- Evaluation: The evaluation criteria are defined by the course tutors and professors. The decisions made by them regarding the results, validity, and any concerns during the competition are final and binding.
- Discretion: Course tutors and professors reserve the right to make decisions regarding any aspect of the competition, including but not limited to the evaluation process and the interpretation of rules.
- Integrity: Participants are expected to maintain academic integrity and professionalism throughout the competition.
By participating, entrants agree to these terms and acknowledge that all decisions by the course tutors and professors are final.