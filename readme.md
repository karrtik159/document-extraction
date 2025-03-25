# Document Extraction using OpenCV

This project demonstrates how to use OpenCV for document extraction, specifically for extracting information from Aadhar and PAN cards. The extraction process uses image processing techniques to identify and extract structured data from these documents.

## Prerequisites

Before you begin, ensure that you have met the following requirements:

- Python 3.x installed on your machine
- OpenCV library installed (`opencv-python`)
- prefer execution on Colab.
## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/document-extraction.git
   cd document-extraction
   ```

2. Install the required libraries:

   ```bash
    !pip install transformers_stream_generator
    !pip install transformers
    !pip install sentencepiece
    !pip install gradio
    !pip install pdf2image
    !pip install pytesseract
    !pip install deepface
    !pip install qwen-vl-utils[decord]
   ```

## Usage


1. Run the extraction script:

   ```bash
   python app.py
   ```
2. Place the images of Aadhar and PAN cards in the simultaneously in gradio.

3. The script will process the images and output the extracted information in a structured format.

## Example

The output will be in JSON format, containing extracted fields such as:

- **Aadhar Card** or  **PAN Card**:
  ```json
    {
        "PAN Found":[0:"BPPCS6267D"],
        "Aadhaar Found":[]
    }
  ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

If you have any questions or suggestions, please open an issue or contact me at [karrtikbaheti159@gmail.com].
