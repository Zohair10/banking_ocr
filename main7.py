import streamlit as st
import base64
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF
import docx2txt  # For extracting images from DOCX
import tempfile
import json
from difflib import SequenceMatcher
from PIL import Image
import io
import requests
from pymongo import MongoClient
# Load environment variables from .env file
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MONGODB_ATLAS_URI = os.getenv("MONGODB_ATLAS_URI")

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

# Send base64 image to  API for JSON extraction
def extract_text_from_image(base64_image):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/llama-4-scout",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": '''<role>You are an expert OCR and form data extraction specialist.</role>
<task>
From this image, determine if it belongs to a **Merchant Application Form**. 
Only if it is a valid Merchant Application Form, extract **all visible information** from it and return it as a valid JSON object with key-value pairs. 

If the image does **not** contain a Merchant Application Form, return exactly this JSON:
{"form_type": "not_merchant_form"}
</task>

<form_identification>
To identify a valid Merchant Application Form, look for keywords such as:
- "Merchant Application Form"
- "Merchant Name", "Business Address", "Account", "NIC", "Legal Structure", etc.
- Typical form layout with labeled fields and handwritten or typed responses
If these are not present, treat it as an irrelevant form or image.
</form_identification>

<instructions>
1. ONLY return data if the image contains a merchant application form
2. If it's a merchant application form, extract every visible field from it
3. Your response MUST be a valid JSON object only — no markdown, no text before or after
4. Use these **exact key names** when matching fields:
- "Date"
- "MID"
- "TID"
- "New Outlet"
- "Chain Outlet"
- "Merchant Name Commercial"
- "Merchant Name legal"
- "Established Since"
- "Business Address Commercial"
- "City"
- "Telephone / Cell"
- "Email/Web"
- "Contact Person Name"
- "Business Address Legal"
- "Number of Outlets"
- "Location of Branches"
- "Type of Business/Type of Merchandise/Service Sold"
- "Annual Sales Volume"
- "Average Transaction size"
- "Expected Volume"
- "Legal Structure"
- "First Name"
- "Last Name"
- "NIC (Old)"
- "NIC New"
- "Residence Address"
- "Authorized Signatory First Name"
- "Authorized Signatory Last Name"
- "Authorized Signatory NIC(Old)"
- "Authorized Signatory NIC(New)"
- "Payment Mode"
- "Banker Name & Branch"
- "Account/IBAN"
- "Merchant Cheaque Beneficiary Name"
- "Merchant Cheaque Beneficiary Address"
- "Do You want Direct Credit Facility with UBL"
- "If any previous Credit Card acceptance relationship"
- "If yes, with"
- "Current Status of Relationship"
- "If active, what equipment is already in place"
- "If Terminated Reason of Termination"
- "Discount Rates Offered"
If a field is missing or unreadable, return its value as `null`.
</instructions>

<formatting>
- NEVER use markdown or backticks
- ALWAYS return a plain JSON object
- ALWAYS ensure valid JSON syntax with double quotes
- NEVER hallucinate or assume field values
- ALWAYS use `null` if uncertain
</formatting>
<examples>
Good response: {"Date": "2025-07-09", "Merchant Name Commercial": "ABC Store", "Telephone": "1234567890", "Business Address Commercial": "123 Main St", "City": "Karachi", "Anual Sales Volume": "100000", "Average Transaction size": "5000"}
Bad response: json {"Date": "2025-07-09"} or "Here is the extracted data: {..."
</examples>
'''
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0,
        "max_tokens": 1024
    }
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        if '{' in content:
            content = content[content.find('{'):]
        if '}' in content:
            content = content[:content.rfind('}') + 1]
        json.loads(content)
        return content
    except Exception as e:
        return '{"error": "Could not extract valid JSON from image or API error."}'
def match_and_autofill_fields(extracted_json):
    """
    Given the extracted JSON, return a dict with the required keys auto-populated if possible.
    Matching is case-insensitive and ignores minor variations.
    """
    required_keys = [
        "Date",
        "MID",
        "TID",
        "New Outlet",
        "Chain Outlet",
        "Merchant Name Commercial",
        "Merchant Name legal",
        "Established Since",
        "Business Address Commercial",
        "City",
        "Telephone / Cell",
        "Email/Web",
        "Contact Person Name",
        "Business Address Legal",
        "Number of Outlets",
        "Location of Branches",
        "Type of Business/Type of Merchandise/Service Sold",
        "Annual Sales Volume",
        "Average Transaction size",
        "Expected Volume",
        "Legal Structure",
        "First Name",
        "Last Name",
        "NIC (Old)",
        "NIC New",
        "Residence Address"
        "Authorized Signatory First Name",
        "Authorized Signatory Last Name",
        "Authorized Signatory NIC(Old)",
        "Authorized Signatory NIC(New)",
        "Payment Mode",
        "Banker Name & Branch",
        "Account/IBAN",
        "Merchant Cheaque Beneficiary Name",
        "Merchant Cheaque Beneficiary Address",
        "Do You want Direct Credit Facility with UBL",
        "If any previous Credit Card acceptance relationship",
        "If yes, with",
        "Current Status of Relationship",
        "If active, what equipment is already in place",
        "If Terminated Reason of Termination",
        "Discount Rates Offered"
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

def get_mongo_collection():
    """
    Get MongoDB collection using Atlas connection with fallback to local.
    Returns None if connection fails.
    """
    try:
        # Try MongoDB Atlas first
        if MONGODB_ATLAS_URI and MONGODB_ATLAS_URI != "mongodb+srv://<username>:<password>@<cluster-url>/<database>?retryWrites=true&w=majority":
            client = MongoClient(MONGODB_ATLAS_URI, serverSelectionTimeoutMS=5000)
            # Test the connection
            client.admin.command('ping')
            db = client["formextract_db"]
            collection = db["submitted_forms"]
            return collection
        else:
            # Fallback to local MongoDB
            st.warning("MongoDB Atlas URI not configured. Using local MongoDB connection.")
            client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            # Test the connection
            client.admin.command('ping')
            db = client["formextract_db"]
            collection = db["submitted_forms"]
            return collection
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

def main():
    st.title("FormExtract AI")
    st.markdown("Upload an **image, PDF, or DOCX** file containing handwritten text to extract structured data using AI.")
    required_keys = [
        "Date",
        "MID",
        "TID",
        "New Outlet",
        "Chain Outlet",
        "Merchant Name Commercial",
        "Merchant Name legal",
        "Established Since",
        "Business Address Commercial",
        "City",
        "Telephone / Cell",
        "Email/Web",
        "Contact Person Name",
        "Business Address Legal",
        "Number of Outlets",
        "Location of Branches",
        "Type of Business/Type of Merchandise/Service Sold",
        "Annual Sales Volume",
        "Average Transaction size",
        "Expected Volume",
        "Legal Structure",
        "First Name",
        "Last Name",
        "NIC (Old)",
        "NIC New",
        "Residence Address",
        "Authorized Signatory First Name",
        "Authorized Signatory Last Name",
        "Authorized Signatory NIC(Old)",
        "Authorized Signatory NIC(New)",
        "Payment Mode",
        "Banker Name & Branch",
        "Account/IBAN",
        "Merchant Cheaque Beneficiary Name",
        "Merchant Cheaque Beneficiary Address",
        "Do You want Direct Credit Facility with UBL",
        "If any previous Credit Card acceptance relationship",
        "If yes, with",
        "Current Status of Relationship",
        "If active, what equipment is already in place",
        "If Terminated Reason of Termination",
        "Discount Rates Offered"
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
            with st.spinner(":arrows_counterclockwise: Processing all images and extracting data..."):
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
            st.subheader(":link: Final Extracted Data")
            st.json(combined_json)
            st.success(":white_check_mark: Extraction complete! Check the form below.")
        else:
            st.info("Click 'Extract Data' button above to start processing the uploaded file.")
    # --- Form section below ---
    st.markdown("---")
    st.header("Form Details")
    # Show status
    if st.session_state.extraction_complete:
        st.info(":clipboard: Form auto-filled with extracted data. You can edit the fields before submitting.")
    else:
        st.info(":clipboard: Fill out the form manually or upload a file above to auto-fill.")
    # The form with auto-filled or empty values
    with st.form("data_form"):
        form_values = {}
        for key in required_keys:
            current_value = st.session_state.extracted_autofill.get(key, "")
            form_values[key] = st.text_input(
                label=key,
                value="" if current_value is None else str(current_value),
                help=f"Enter {key}" if not st.session_state.extraction_complete else f"Auto-filled from extraction",
                key=f"input_{key}"
            )
        # Submit button
        submitted = st.form_submit_button(":outbox_tray: Submit Form", use_container_width=True)
        if submitted:
            # Store in MongoDB
            collection = get_mongo_collection()
            if collection is not None:
                try:
                    insert_result = collection.insert_one(form_values)
                    st.success("✅ Form submitted successfully and saved to database!")
                    st.subheader("📋 Submitted Data:")
                    st.json(form_values)
                    st.info(f"Form saved with MongoDB ID: {insert_result.inserted_id}")
                except Exception as e:
                    st.error(f"Failed to save form to database: {str(e)}")
                    st.subheader("📋 Form Data (Not Saved):")
                    st.json(form_values)
            else:
                st.error("❌ Database connection failed. Form data not saved.")
                st.subheader("📋 Form Data (Not Saved):")
                st.json(form_values)
if __name__ == "__main__":
    main()
