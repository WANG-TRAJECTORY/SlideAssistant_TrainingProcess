import pdf2image
import pytesseract
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

pdf_files = [
    r'C:\Users\wtupw\Desktop\slideassistant_slides\CUC_Deeplearning_allslides.pdf',
    r'C:\Users\wtupw\Desktop\slideassistant_slides\d2l-zh-pytorch.pdf'
]

output_dir = r'C:\Users\wtupw\Desktop\slideassistant_slides'
os.makedirs(output_dir, exist_ok=True)

for pdf_path in pdf_files:
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        continue

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_text_file = os.path.join(output_dir, f'{pdf_name}_extracted_text.txt')

    try:
        images = pdf2image.convert_from_path(pdf_path)
    except Exception as e:
        print(f"Error converting PDF {pdf_path} to images: {e}")
        continue

    with open(output_text_file, 'w', encoding='utf-8') as text_file:
        for i, image in enumerate(images):
            print(f"Processing page {i + 1}/{len(images)} of {pdf_name}")

            try:
                text = pytesseract.image_to_string(image, lang='eng+chi_sim')
                text_file.write(f'\n=== Page {i + 1} ===\n')
                text_file.write(text)
            except Exception as e:
                print(f"Error processing page {i + 1} of {pdf_name}: {e}")

    print(f"Text extraction completed for {pdf_name}. Output saved to {output_text_file}")

print("All PDFs processed.")