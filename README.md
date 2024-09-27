# 🚢 Container Number Extraction Project 📦

## 🔍 Overview

The **Container Number Extraction Project** aims to automate the extraction of container numbers from various image sources using cutting-edge image recognition and OCR (Optical Character Recognition) technologies. Our goal is to create a robust, accurate, and user-friendly system that handles different image qualities, angles, and lighting conditions efficiently. 

## ✨ Key Features

- 🖼️ **Image Preprocessing**: Automatically enhances image quality with noise reduction, resizing, and contrast adjustments.
- 🔠 **Advanced OCR**: Integrates state-of-the-art OCR models to detect and accurately extract container numbers.
- 🧠 **Custom Algorithm**: Fine-tuned algorithms to maximize accuracy based on container number formats.
- ⚠️ **Error Handling**: Built-in error detection and recovery for images with unclear or partially visible numbers.
- 💻 **User Interface**: Simple and intuitive UI for easy image uploads and data viewing.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/container-number-extraction.git
   pip install -r requirements.txt
2. **Choose `CNTR_Size.ipynb`**:
   - This notebook handles the detection of container numbers.
   - Sample images will be processed using the implemented logic, and the container numbers will be automatically detected and extracted.
   - Results will be displayed, including the detected container numbers, along with confidence scores and the status (success/fail).

3. **Processing**: 
   - Once uploaded, the system preprocesses images to optimize clarity by adjusting brightness, contrast, and reducing noise.
   - The system ensures the image is resized appropriately for the OCR engine.

4. **Extract**:
   - The OCR model extracts container numbers from the preprocessed image. 
   - Detected numbers are displayed in the output view, with confidence scores and a success/fail status for each image.

5. **Save & Export**:
   - Users can save the results in various formats (CSV, JSON) for further analysis or integration into existing logistics systems.
  
## Mathematical Approach to Candidate Selection

The process of identifying container number candidates and arranging contours follows a structured approach, using the following techniques:

### Mathematical Formula for Aspect Ratio
We calculate the aspect ratio of contours to filter out non-container-like candidates. The formula is:

$$
Aspect\ Ratio = \frac{Width\ of\ Contour}{Height\ of\ Contour}
$$

This ratio helps in eliminating regions that do not fit the expected width-to-height ratio of a container number.

### Area Filtering
Contours are filtered based on the area to ensure we focus on regions large enough to contain valid numbers. The area is calculated as:

$$
Area = Width \times Height
$$

Only contours within a reasonable size range are kept for further evaluation.

### Contour Sorting and Arrangement
Once the contours are filtered, we arrange them based on their position within the image, ensuring that the container number is read from left to right. This is achieved by sorting the contours along the x-axis. The following formula is used:

$$
Contour\ Arrangement = sort(x\ coordinates)
$$

This ensures that the characters are in the correct order for OCR recognition.

## Results
After processing, the detected container numbers will be displayed in the notebook, both visually (through highlighted contours on the image) and textually (as the OCR-extracted text).

## Contact Information
For any inquiries or collaborations, feel free to reach out to me:

- **Email**: songjeongjun320@gmail.com
- **LinkedIn**: [Your LinkedIn Profile](www.linkedin.com/in/junsong0602)
- **Portfolio**: [Your portfolio](https://songjeongjun320.github.io/)
