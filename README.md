# SlideAssistant_TrainingProcess
Author: Ting Wang（王霆）
## Project Overview
SlideAssistant is a Python-based project designed to assist with deep learning and machine learning education by processing course slide content, generating relevant questions, and providing accurate answers using a fine-tuned Qwen2-0.5B model. The project integrates multiple components, including PDF text extraction, question generation, model fine-tuning with LoRA, and semantic search for context-aware question answering.
Features

PDF Text Extraction: Extracts text from PDF slides using pdf2image and pytesseract, supporting both English and Chinese.
Question Generation: Generates a comprehensive set of deep learning and machine learning questions using the DeepseekR1 model.
Model Fine-Tuning: Fine-tunes the Qwen2-0.5B model with LoRA distillation and unsupervised pretraining on slide content.
Semantic Search: Uses the SentenceTransformer model (all-MiniLM-L6-v2) to retrieve relevant slide content for answering user queries.
Context-Aware Question Answering: Combines semantic search with the fine-tuned Qwen model to provide accurate answers with minimal reliance on context.

## Requirements
To run this project, ensure you have the following dependencies installed:
pip install torch transformers sentence-transformers pdf2image pytesseract datasets peft openai

## Additional requirements:

Tesseract-OCR: Install Tesseract-OCR and set the path in the script (e.g., pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe').
Poppler: Required for pdf2image. Install it and add it to your system PATH.
CUDA-enabled GPU (optional): For faster model inference and training.



## Setup Instructions

### Clone the Repository:
git clone https://github.com/WangTrajectory/SlideAssistant_TrainingProcess .git
cd SlideAssistant


### Install Dependencies:
pip install -r requirements.txt


### Prepare PDF Slides:

Place your PDF slide files in a directory (e.g., C:\Users\YourName\Desktop\slideassistant_slides\).
Update the pdf_files list in script2.py with the correct paths to your PDFs.


### Set API Key for DeepseekR1:

Obtain an API key for DeepseekR1 and update the api_key variable in script3.py.



## Usage

Extract Text from Slides:
python script2.py

This processes PDF slides and saves the extracted text to course_slides_text.txt.

Generate Questions and Answers:
python script3.py

This generates a list of deep learning questions and answers using DeepseekR1, saved to deepseek_responses.txt.

Fine-Tune the Qwen Model:
python script4.py

This fine-tunes the Qwen2-0.5B model using LoRA and unsupervised PFT, saving the results to lora_finetuned/ and LoRA&pft_finetuned/.

Answer Questions:
python script5.py

This loads the fine-tuned model, performs semantic search on slide content, and answers user queries (e.g., "什么是padding?").



## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Qwen for the lightweight language model.
Hugging Face Transformers for model training and inference.
Sentence Transformers for semantic embeddings.
DeepseekR1 for question generation.
