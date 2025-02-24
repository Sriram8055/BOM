#current one
import streamlit as st
# from google.generative_language import Client, GenerateTextRequest
import google.generativeai as genai
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import re
import time
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Configure API key (replace with your actual API key)
GOOGLE_API_KEY =  os.getenv('GEMINI_API')
genai.configure(api_key=GOOGLE_API_KEY)

# Image Preprocessing Function
def preprocess_image(image):
    """
    Enhances the image so that black lines become darker and gray areas become more distinct.
    """
    # 1. Convert to Grayscale
    img_gray = np.array(image.convert("L"))  # Convert PIL Image to NumPy array

    # 2. Apply CLAHE (Local Contrast Enhancement)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)

    # 3. Bilateral Filtering (Preserve Edges, Reduce Noise)
    img_bilateral = cv2.bilateralFilter(img_clahe, d=9, sigmaColor=75, sigmaSpace=75)

    # 4. Increase Contrast (Makes black lines darker, gray areas more distinct)
    #    alpha > 1.0 boosts contrast, beta shifts brightness
    alpha = 1.1  # Increase for stronger contrast
    beta = +90   # Slightly darken the overall image
    img_contrast = cv2.convertScaleAbs(img_bilateral, alpha=alpha, beta=beta)

    # 5. Morphological Close (Fills small gaps, strengthens lines)
    #    This helps unify broken or thin lines into solid shapes.
    kernel = np.ones((3, 3), np.uint8)
    img_morph = cv2.morphologyEx(img_contrast, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(img_morph)

# Function to Chunk Image
def chunk_image(image, num_chunks_x, num_chunks_y):
    """
    Splits an image into equal non-overlapping chunks both horizontally and vertically.

    Args:
        image (PIL.Image): Input image.
        num_chunks_x (int): Number of chunks along the width.
        num_chunks_y (int): Number of chunks along the height.

    Returns:
        list: A list of cropped image chunks.
    """
    width, height = image.size
    chunk_width = width // num_chunks_x  # Base width of each chunk
    chunk_height = height // num_chunks_y  # Base height of each chunk
    
    remainder_x = width % num_chunks_x  # Handle cases where width isn't evenly divisible
    remainder_y = height % num_chunks_y  # Handle cases where height isn't evenly divisible
    
    chunks = []
    
    top = 0
    for j in range(num_chunks_y):
        extra_pixel_y = 1 if j < remainder_y else 0  # Distribute extra pixels across rows
        bottom = top + chunk_height + extra_pixel_y
        
        left = 0
        for i in range(num_chunks_x):
            extra_pixel_x = 1 if i < remainder_x else 0  # Distribute extra pixels across columns
            right = left + chunk_width + extra_pixel_x
            
            chunk = image.crop((left, top, right, bottom))
            chunks.append(chunk)

            left = right  # Move to next column

        top = bottom  # Move to next row
    
    return chunks



# Function to Extract Symbol Counts using Regex
def extract_counts(text):
    counts = {
        "Ceiling Diffusers": 0,
        "Ceiling Return Air Grilles": 0,
        "Thermostats": 0
    }
    patterns = {
        "Ceiling Diffusers": r"Ceiling Diffusers:\s*(\d+)",
        "Ceiling Return Air Grilles": r"Ceiling Return Air Grilles:\s*(\d+)",
        "Thermostats": r"Thermostats:\s*(\d+)",
        "Hex Labels": r"Hex Labels:\s*(\d+)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            counts[key] = int(match.group(1))
    
    return counts

# Gemini Analysis Function
def analyze_image_chunk(image_chunk):
    prompt = """
### **Mechanical Diagram Symbol Detection & Counting Task**

#### **Role:**  
You are an expert in **mechanical blueprint analysis**.  
Your ONLY task is to **identify and count symbols** in the given **technical drawing** with **100% accuracy** by following the strict rules below.

---

### **Symbols to Identify & Count:**

#### **1. Ceiling Diffuser (CD)**
- **Appearance:** A **small, fully enclosed perfect square** with a **bold, dark 'X' shape** inside.
- **Strict Validation Rules:**  
  ✅ The symbol must display **EXACTLY TWO distinct diagonal lines** that intersect **precisely at the center** to form a perfect “X.”  
  ✅ One diagonal must run from **top-left to bottom-right** and the other from **top-right to bottom-left**, with both lines extending **fully from corner to corner** and dividing the square into **four equal right-angled triangles**.  
  ✅ Both diagonal lines must be **uniformly thick, fully opaque dark black, and identical in width** throughout (no fading, thinning, or color variation).  
  ✅ The square **MUST be a perfect square** with equal side lengths and right angles at all four corners.  
  ✅ The square **MUST be unfilled** (pure white/transparent interior) with a **solid, continuous black border** (the border must be completely black—not gray, faded, dashed, or broken).  
  ✅ The symbol must be of **standard size** (neither oversized nor undersized) and **completely visible** (not cut off, overlapped, or partially obscured).  
  ✅ **No additional markings, gradients, shadows, or extra lines** should appear inside or around the square.
  ✅ **If only one diagonal line is present or if the 'X' is not perfectly symmetric, misaligned, or uneven in thickness, it must not be counted as a Ceiling Diffuser.**

- **DO NOT count if:**  
  ❌ The square is **not perfectly formed** (e.g., rectangular, distorted, or with uneven sides).  
  ❌ There is **only ONE diagonal line** or if the two diagonal lines do not form a clear, perfectly aligned “X.”  
  ❌ The “X” is **incomplete, irregular, misaligned, uneven in thickness, or unclear** in any way.  
  ❌ Any diagonal line appears **faded, partially broken, or not fully black.**  
  ❌ The square is **filled, shaded, hatched**, or contains **any internal noise** (the interior must be completely white/transparent).  
  ❌ The border is **not a solid, continuous black line** (if it is gray, dashed, broken, or otherwise unclear, do not count it).  
  ❌ The shape is **oversized, undersized, partially visible, overlapped**, or has **background lines crossing it**.  
  ❌ There are **any extra lines, text, or markings** inside the square that interfere with the clarity and perfection of the “X.”

#### **2. Ceiling Return Air Grille (RA)**
- **Appearance:** A **small, perfectly enclosed square** (equal width & height) with **one bold diagonal** from **bottom-left** to **top-right**.
- **Strict Validation Rules:**  
  ✅ Must have a **solid black outline** forming **four distinct 90° corners** (a perfect square).  
  ✅ The interior must be **white/transparent** (no shading, coloring, or hatching).  
  ✅ Must have **exactly one** diagonal line running corner-to-corner (bottom-left → top-right).  
  ✅ The diagonal line must be **thick, dark, well-defined**, and completely contained within the square.  
  ✅ The symbol must be **fully visible**, of **standard symbol size**, and **not overlapped** by other lines.  
  ✅ **No extra or repeated diagonal lines, dashed edges, or hatching** should be present.
  
  **DO NOT count if:**  
  ❌ The shape is **rectangular** (not a perfect square) or is **open/dashed**.  
  ❌ The interior is **gray, shaded, hatched, or colored**.  
  ❌ There are **multiple diagonals** or the diagonal is in the **wrong direction** (e.g., top-left → bottom-right).  
  ❌ Two diagonals form an “X” (which qualifies as a diffuser).  
  ❌ The diagonal is **faint, incomplete, or extends outside** the square’s corners.  
  ❌ The shape is **large, overlapped, partially visible**, or consists only of **horizontal/vertical lines** without a closed square.  
  ❌ **Any extra lines, arrows, or repeated diagonal patterns** pass through or behind the square.  
  ❌ The shape is larger than a **typical grille symbol** or is part of a **hatched/filled region**.

#### **3. Thermostat (THER)**
- **Appearance:** A **small, fully enclosed circle** with a **capital letter 'T'** clearly centered inside.
- **Strict Validation Rules:**  
  ✅ The **circle must have a solid outline**, fully closed (no breaks or distortions).  
  ✅ The **letter 'T'** must be **bold, dark, well-defined**, and perfectly **centered** within the circle.  
  ✅ The symbol must be of **standard size** (not oversized or undersized) and **fully visible** (not cut off, overlapped, or partially obscured).
  
  **DO NOT count if:**  
  ❌ The **'T' is missing, faint, or unclear**.  
  ❌ The **circle is incomplete, broken, or irregular**.  
  ❌ The symbol is **cut off, overlapped, or partially visible** at the image edge.  
  ❌ There are **extra markings or fill** inside the circle beyond the 'T'.

#### **4. Hex Label (HX)**
- **Appearance:** A **small, fully enclosed hexagon** (six straight edges, all sides equal) with a **bold, dark digit** (e.g., “5”) clearly centered inside.
- **Strict Validation Rules:**  
  ✅ The **hexagon** must have **six distinct edges** forming a regular (or near-regular) hexagon with six consistent angles.  
  ✅ The interior must be **white/transparent** (no shading, coloring, or hatching).  
  ✅ The **digit** must be **dark, well-defined, and clearly centered** within the hexagon.  
  ✅ The symbol must be of **standard size** (neither oversized nor undersized) and **fully visible** (not cut off or overlapped).  
  ✅ The hexagon’s outline must be **solid black** (not gray, faded, dashed, or broken).  
  ✅ **No additional markings, gradients, shadows, or extra lines** should appear inside or around the hexagon.

  **DO NOT count if:**  
  ❌ The shape is **not a proper hexagon** (missing sides, irregular edges, or angles).  
  ❌ The digit is **faint, unclear, or missing**.  
  ❌ The outline is **not solid black** (gray, dashed, broken, or unclear).  
  ❌ The hexagon is **filled, shaded, hatched**, or has **internal noise**.  
  ❌ The shape is **oversized, undersized, partially visible**, or overlapped by other elements.  
  ❌ Any extra lines, text, or markings inside the hexagon interfere with the clarity of the digit.

---

### **Common Mistakes to Avoid:**
- **Ceiling Diffusers** MUST have **exactly TWO diagonal lines** forming a **perfect 'X'**; if only one diagonal is present, it is NOT a diffuser.  
- **Ceiling Return Air Grilles** MUST have a **single diagonal line** in a **perfect square** with no additional or repeated lines.  
- **Thermostats** and **Hex Labels** must be clear and isolated, with no extra marks inside.  
- **Reject** any symbol that is **filled, overlapped, hatched, or has additional lines/markings** that compromise clarity.  
- If **uncertain**, list the symbol under “Unclear Symbols” rather than guessing.

---

### **Strict Rules for Symbol Counting:**

#### **1. Step-by-Step Counting Method:**
1. **Scan left to right, top to bottom** thoroughly—every nook and corner must be checked.  
2. **Confirm** each symbol is a **fully enclosed, small shape** (perfect square for diffusers/return air grilles, perfect circle for thermostats, perfect hexagon for hex labels) with the correct internal features.  
3. **Reject** any symbol that does not meet the criteria (e.g., if the “X” is imperfect or only one diagonal is present, do not count it as a Ceiling Diffuser).

#### **2. Confidence Score & Unclear Cases:**
- **Assign a confidence score (0-100%)** for each symbol.
- **Only count symbols if the confidence score is 100%**; if the score is below 100%, place the symbol in **“Unclear Symbols”** with a reason.

---

### **Final Response Format (Must Follow Exactly):**
```plaintext
Ceiling Diffusers: [Exact Count] (Confidence: XX%)
Ceiling Return Air Grilles: [Exact Count] (Confidence: XX%)
Thermostats: [Exact Count] (Confidence: XX%)
Hex Labels: [Exact Count] (Confidence: XX%)
Unclear Symbols:
- **Description:** [Mention why unclear]
- **Estimated Count (if applicable):** [Only if partially valid]
- **Confidence Score:** XX%


Double-check your response before finalizing!
"""

    
    # Convert image chunk to bytes
    buffered = BytesIO()
    image_chunk.save(buffered, format="PNG")
    image_data = buffered.getvalue()
    generation_config = {
    "temperature": 0.01,  # Still low to maintain accuracy
    "top_p": 0.30,  # Allows more diversity
    "top_k": 40,  # Ensures only top-ranked completions are considered
    "max_output_tokens": 500,  # Adjusted to avoid unnecessary text
    "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel("gemini-1.5-flash",generation_config=generation_config)
    response = model.generate_content([prompt, {"mime_type": "image/png", "data": image_data}])
    print(response.text.strip())
    return response.text.strip()
import fitz
# Streamlit UI
st.title("Mechanical Diagram Symbol Detection")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    
    # Ensure the PDF has at least 4 pages
    if len(pdf_document) < 4:
        st.error("The uploaded PDF has less than 4 pages. Please upload a longer PDF.")
    else:
        # Select the 4th page (index 3, as index starts from 0)
        page = pdf_document[3]

        # Convert PDF page to high-resolution image
        dpi = 300  # High-resolution DPI
        matrix = fitz.Matrix(dpi / 72, dpi / 72)  # Scale factor
        pix = page.get_pixmap(matrix=matrix)

        # Convert to PIL image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    image = img
    st.write("## Original Image")
    st.image(image, caption=f"Processing image...",  use_container_width=True)
    processed_image = preprocess_image(image)
    st.write("## Preprocessed Image")
    st.image(processed_image, caption=f"Processing image...",  use_container_width=True)
    num_chunks_x = 6
    num_chunks_y = 4
    if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                chunks = chunk_image(processed_image, num_chunks_x, num_chunks_y)

                total_counts = {
                    "Ceiling Diffusers": 0,
                    "Ceiling Return Air Grilles": 0,
                    "Thermostats": 0,
                    "Hex Labels": 0  # New category for hexagon labels
                }
                progress_bar = st.progress(0)

                for i, chunk in enumerate(chunks):
                    print("-" * 50)
                    print(f"Processing Chunk {i+1}")
                    # st.image(chunk, caption=f"Processing Chunk {i+1}...", use_column_width=True)
                    
                    response_text = analyze_image_chunk(chunk)
                    counts = extract_counts(response_text)

                    # Update total counts
                    for key in total_counts:
                        total_counts[key] += counts.get(key, 0)  # Ensure missing keys don't cause errors
                    progress_bar.progress((i + 1) /24)
                    # Display chunk results
                    # st.write(f"### Chunk {i+1} Results:")
                    # st.write(f"**Ceiling Diffusers:** {counts.get('Ceiling Diffusers', 0)}")
                    # # st.write(f"**Ceiling Return Air Grilles:** {counts.get('Ceiling Return Air Grilles', 0)}")
                    # st.write(f"**Thermostats:** {counts.get('Thermostats', 0)}")
                    # st.write(f"**Hex Labels:** {counts.get('Hex Labels', 0)}")  # New hex label count
                    time.sleep(2)

                df_summary = pd.DataFrame([
                    ["Ceiling Diffusers", total_counts["Ceiling Diffusers"]],
                    ["Thermostats", total_counts["Thermostats"]],
                    ["Keynote", total_counts["Hex Labels"]]
                ], columns=["Symbol", "Total Count"])

                # Display the summary table
                st.write("## Final Symbol Counts Across All Chunks:")
                st.table(df_summary)
                print("-" * 150)

