import streamlit as st
import base64
from groq import Groq
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF
import docx2txt  # For extracting images from DOCX
import tempfile
import json
from difflib import SequenceMatcher
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()
# Initialize Groq client with your API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # Reads from .env or system env
# Helper to convert image bytes to base64
def encode_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")
# Extract images from PDF
def extract_images_from_pdf(pdf_bytes):
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                images.append(image_bytes)
    return images

# Extract images from DOCX
def extract_images_from_docx(docx_path):
    images = []
    temp_dir = tempfile.mkdtemp()
    docx2txt.process(docx_path, temp_dir)
    for fname in os.listdir(temp_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            with open(os.path.join(temp_dir, fname), 'rb') as f:
                images.append(f.read())
    return images

# Send base64 image to Groq API for JSON extraction
def extract_text_from_image(base64_image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": '''<role>You are an expert OCR and form data extraction specialist.</role>
<task>
Extract ALL visible information from this form image and return it as a valid JSON object with key-value pairs.
If you find any of the following fields, use these exact key names in your JSON:
- "Date"
- "Merchant Name Commercial"
- "Merchant Name legal"
- "Business Address Commercial"
- "City"
- "Telephone"
- "Anual Sales Volume"
- "Average Transaction size"
- "Legal Structure"
- "First Name"
- "Last Name"
- "NIC New"
- "Payment Mode"
- "Banker Name and Branch"
- "Account"
If a field is not present, set its value to null.
</task>
<instructions>
1. MANDATORY: Your response MUST be a valid JSON object only - no additional text, explanations, or formatting
2. Extract both handwritten and printed text accurately
3. For checkboxes/tick marks: identify selected options and include them as boolean values
4. For empty/blank fields: use null as the value
5. Preserve exact formatting for phone numbers, dates, and identification numbers
6. For addresses: capture complete address as single string value
</instructions>
<guardrails>
- NEVER include explanatory text before or after the JSON
- NEVER use markdown code blocks or backticks
- NEVER hallucinate or infer data not visible in the image
- ALWAYS use double quotes for JSON strings
- ALWAYS ensure valid JSON syntax
- IF uncertain about a value, use null instead of guessing
</guardrails>
<examples>
Good response: {"Date": "2025-07-09", "Merchant Name Commercial": "ABC Store", "Telephone": "1234567890", "Business Address Commercial": "123 Main St", "City": "Karachi", "Anual Sales Volume": "100000", "Average Transaction size": "5000"}
Bad response: json {"Date": "2025-07-09"} or "Here is the extracted data: {..."
</examples>
'''} ,
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        temperature=0,
        max_completion_tokens=1024,
    )
    response = completion.choices[0].message.content.strip()
    if '{' in response:
        response = response[response.find('{'):]
    if '}' in response:
        response = response[:response.rfind('}') + 1]
    try:
        json.loads(response)
        return response
    except json.JSONDecodeError:
        return '{"error": "Could not extract valid JSON from image"}'

def match_and_autofill_fields(extracted_json):
    """
    Given the extracted JSON, return a dict with the required keys auto-populated if possible.
    Matching is case-insensitive and ignores minor variations.
    """
    required_keys = [
        "Date",
        "Merchant Name Commercial",
        "Merchant Name legal", 
        "Business Address Commercial",
        "City",
        "Telephone",
        "Anual Sales Volume",
        "Average Transaction size",
        "Legal Structure",
        "First Name",
        "Last Name",
        "NIC New",
        "Payment Mode",
        "Banker Name and Branch",
        "Account"
    ]
    autofill = {k: None for k in required_keys}
    if not extracted_json:
        return autofill
    for req_key in required_keys:
        for k, v in extracted_json.items():
            # Simple normalization for matching
            if req_key.lower().replace(" ", "").replace("(", "").replace(")", "").replace("+", "") in k.lower().replace(" ", "").replace("(", "").replace(")", "").replace("+", ""):
                autofill[req_key] = v
                break
    return autofill

def char_similarity(a, b):
    return SequenceMatcher(None, a or "", b or "").ratio()

def calculate_accuracy(extracted, corrected):
    total_fields = len(corrected)
    exact_matches = 0
    char_scores = {}
    for key in corrected:
        extracted_val = (extracted.get(key) or "").strip() if extracted.get(key) is not None else ""
        corrected_val = (corrected.get(key) or "").strip() if corrected.get(key) is not None else ""
        if extracted_val == corrected_val:
            exact_matches += 1
        similarity = char_similarity(extracted_val, corrected_val)
        char_scores[key] = round(similarity * 100, 2)
    field_accuracy = round((exact_matches / total_fields) * 100, 2)
    average_char_accuracy = round(sum(char_scores.values()) / total_fields, 2)
    return field_accuracy, average_char_accuracy, char_scores

def main():
    st.title("FormExtract AI")
    st.markdown("Upload an **image, PDF, or DOCX** file containing handwritten text to extract structured data using AI.")
    
    required_keys = [
        "Date",
        "Merchant Name Commercial",
        "Merchant Name legal", 
        "Business Address Commercial",
        "City",
        "Telephone",
        "Anual Sales Volume",
        "Average Transaction size",
        "Legal Structure",
        "First Name",
        "Last Name",
        "NIC New",
        "Payment Mode",
        "Banker Name and Branch",
        "Account"
    ]
    
    # Initialize session state for extracted data
    if 'extracted_autofill' not in st.session_state:
        st.session_state.extracted_autofill = {k: "" for k in required_keys}
    if 'extraction_complete' not in st.session_state:
        st.session_state.extraction_complete = False
    
    # --- File upload and extraction logic at the top ---
    st.header(":file_folder: File Upload & Extraction")
    uploaded_file = st.file_uploader("Upload File", type=["jpg", "jpeg", "png", "pdf", "docx"])
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Add Extract button
        extract_button = st.button("Extract Data", use_container_width=True, type="primary")
        
        if extract_button:
            file_type = uploaded_file.type
            file_bytes = uploaded_file.read()
            
            # Determine file type and extract images
            if file_type.startswith("image/"):
                images = [file_bytes]
            elif uploaded_file.name.endswith(".pdf"):
                images = extract_images_from_pdf(file_bytes)
            elif uploaded_file.name.endswith(".docx"):
                temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(file_bytes)
                images = extract_images_from_docx(temp_path)
            else:
                st.warning("Unsupported file type.")
                return
            
            if not images:
                st.warning("No images found in the uploaded file.")
                return
            
            st.success(f":white_check_mark: Found {len(images)} image(s). Processing now...")
            
            # Initialize session state for all extracted data
            if 'all_extracted_data' not in st.session_state:
                st.session_state.all_extracted_data = []
            
            # Single loading indicator for the entire process
            with st.spinner("🔄 Processing all images and extracting data..."):
                # Combine all extracted data into one JSON object (only update null values)
                combined_json = {}
                
                for i, img_bytes in enumerate(images):
                    base64_img = encode_image_to_base64(img_bytes)
                    
                    result = extract_text_from_image(base64_img)
                    try:
                        json_obj = json.loads(result)
                        st.session_state.all_extracted_data.append(json_obj)
                        
                        # Only update fields that are null/empty in combined_json
                        for key, value in json_obj.items():
                            if key not in combined_json or combined_json[key] is None or combined_json[key] == "":
                                if value is not None and value != "":
                                    combined_json[key] = value
                    except json.JSONDecodeError:
                        # Skip invalid JSON responses
                        continue
            
            # Auto-populate the required fields from the combined JSON
            autofill = match_and_autofill_fields(combined_json)
            st.session_state.extracted_autofill = autofill
            st.session_state.extraction_complete = True
            
            # Show only the final combined result
            st.markdown("---")
            st.subheader("🔗 Final Extracted Data")
            st.json(combined_json)
            st.success("✅ Extraction complete! Check the form below.")
        else:
            st.info("Click 'Extract Data' button above to start processing the uploaded file.")
    
    # --- Form section below ---
    st.markdown("---")
    st.header("Form Details")
    
    # Show status
    if st.session_state.extraction_complete:
        st.info("📋 Form auto-filled with extracted data. You can edit the fields before submitting.")
    else:
        st.info("📋 Fill out the form manually or upload a file above to auto-fill.")
    
    # The form with auto-filled or empty values
    with st.form("data_form"):
        form_values = {}
        for key in required_keys:
            current_value = st.session_state.extracted_autofill.get(key, "")
            form_values[key] = st.text_input(
                key, 
                value="" if current_value is None else str(current_value),
                help=f"Enter {key}" if not st.session_state.extraction_complete else f"Auto-filled from extraction"
            )
        
        # Submit button
        submitted = st.form_submit_button("📤 Submit Form", use_container_width=True)
        
        if submitted:
            st.success("✅ Form submitted successfully!")
            st.subheader("📋 Submitted Data:")
            st.json(form_values)
            
if __name__ == "__main__":
    main()