import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import io

# Function for OCR processing
def extract_text_from_image(image):
    """
    Extracts text from an image using pytesseract OCR.
    
    Args:
        image (PIL.Image): The input image from which text is to be extracted.
    
    Returns:
        str: The extracted text from the image.
    """
    text = pytesseract.image_to_string(image)
    return text

# Function to map entities to columns based on best match
def map_entities_to_columns(entities, columns):
    """
    Automatically map detected entities to table columns.
    
    Args:
        entities (list): List of detected entities (strings).
        columns (list): List of column names from the table structure.
    
    Returns:
        dict: Mapping of entities to columns.
    """
    entity_to_column = {}
    for i, entity in enumerate(entities):
        # Example simple matching by index for demo purposes
        column = columns[i % len(columns)]  # Simple round-robin assignment
        entity_to_column[entity] = column
    return entity_to_column

# Example table structures
TABLE_STRUCTURES = {
    "Receipts": ["Item", "Price", "Quantity", "Date", "Total"],
    "Invoices": ["Invoice No.", "Date", "Customer", "Amount", "Due Date"]
}

# Streamlit App
st.title("Document Recognition and Table Mapping App")

# Step 1: Select table structure (Receipts/Invoices)
table_type = st.selectbox("Select Table Type", list(TABLE_STRUCTURES.keys()))
columns = TABLE_STRUCTURES[table_type]
st.write(f"Selected Table Structure: {columns}")

# Step 2: Upload or capture an image
image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if image_file is not None:
    # Load the image
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Step 3: Perform OCR to extract text from the image
    st.write("Extracting text from image...")
    extracted_text = extract_text_from_image(image)
    st.write("Detected Entities:")
    st.text(extracted_text)

    # Split detected entities into lines (for simplification)
    detected_entities = extracted_text.splitlines()
    detected_entities = [e for e in detected_entities if e.strip()]  # Remove empty lines
    st.write("Detected Entities List:")
    st.write(detected_entities)

    # Step 4: Auto-map entities to columns
    st.write("Automatically Mapping Detected Entities to Table Columns...")
    entity_column_mapping = map_entities_to_columns(detected_entities, columns)
    st.write(entity_column_mapping)

    # Step 5: Manual adjustment of entity to column mapping
    st.write("Manually Adjust Entity to Column Mapping (if needed):")
    for entity in detected_entities:
        selected_column = st.selectbox(f"Select column for entity '{entity}'", columns, index=columns.index(entity_column_mapping[entity]))
        entity_column_mapping[entity] = selected_column

    # Display final mapping
    st.write("Final Mapping of Entities to Columns:")
    st.write(entity_column_mapping)

    # Step 6: Preview the table with the mapped data
    st.write("Preview Mapped Data Table:")
    preview_data = {column: [] for column in columns}
    for entity, column in entity_column_mapping.items():
        preview_data[column].append(entity)
    
    # Fill empty columns with None or an empty string
    max_len = max(len(values) for values in preview_data.values())
    for column in preview_data:
        while len(preview_data[column]) < max_len:
            preview_data[column].append(None)

    # Create DataFrame to display
    df = pd.DataFrame(preview_data)
    st.write(df)