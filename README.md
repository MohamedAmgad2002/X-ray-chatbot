# Baymax: Medical Chatbot for Diagnostic Insights

**Baymax** is a medical chatbot designed to analyze user-submitted X-ray images and provide diagnostic insights. It also offers real-time conversational capabilities, engaging with patients to address their queries and concerns in a user-friendly manner.

## Features

- **X-ray Image Analysis**
  - Accepts X-ray images from users.
  - Provides diagnostic insights based on the analysis of the submitted images.
  - Covers multiple types of medical images, including chest X-rays, retina scans, kidney images, and brain scans.

- **Real-Time Conversation**
  - Engages users with a conversational interface.
  - Responds to patient queries about medical conditions, symptoms, and X-ray results.
  - Offers user-friendly responses in real-time.

- **Transfer Learning Approach**
  - Uses the Inception and Vgg16 model, optimized through transfer learning, for accurate image classification.
  - Fine-tuned for medical datasets to improve performance across various imaging types.

## Datasets

Baymax works with a combination of different medical image datasets:

1. **Retina Dataset**
   - Contains labeled retinal images for analyzing diabetic retinopathy and Normality.
   - [Link](https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy)

2. **Chest X-ray Dataset**
   - Provides labeled chest X-rays for detecting pneumonia.
   - [Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

3. **Kidney Dataset**
   - Includes kidney images for diagnosing Normal, Cyst, Tumor and Stone.
   - [Link](https://www.kaggle.com/datasets/baalawi1/kidney-diseases-recognition)

4. **Brain Dataset**
   - Contains MRI and CT scans to diagnose brain abnormalities, including tumors.
   - [Link](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MohamedAmgad2002/X-ray-chatbot.git
2. install virtualenv for easier virtual enviroment creation
   ```bash
   pip install virtualenv
3. Create virtual environment
   ```bash
   virtualenv env
   ```
4. Activate environment
  ```bash
  cd \env\Scripts\activate.bat
```
5. Install dependiences
```bash
pip install -r requirements.txt
```
6. Run App in its directory
```bash
chainlit run app.py
```

## How It Works
Upload an X-ray Image:
Users can submit medical images (X-ray, retina scans, etc.) through the interface for diagnostic analysis.

Engage in Conversation:
Users can ask Baymax questions related to their diagnosis, symptoms, or general health concerns. The chatbot responds in real time, providing explanations, guidance, and further assistance.
