import streamlit as st
import requests
import base64
import io
from PIL import Image
import json
import time

# Set page config
st.set_page_config(
    page_title="Face Restoration App",
    layout="wide"
)

st.title("Face Restoration Application")

# Function to test API connectivity
def test_api_connection():
    try:
        response = requests.get('http://localhost:5000/api/test', timeout=5)
        if response.status_code == 200:
            st.success("✅ Connection to backend API successful!")
            return True
        else:
            st.error(f"❌ API test failed with status code: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"❌ Cannot connect to backend API: {str(e)}")
        return False

# Function to process image with backend API
def process_image(image):
    # Convert the image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Debug info
    st.text(f"Image encoded to base64 (length: {len(img_str)} characters)")
    
    # Prepare the payload
    payload = {
        "image": img_str
    }
    
    # Send POST request to API
    status_indicator = st.empty()
    status_indicator.info("Sending request to API...")
    
    try:
        response = requests.post('http://localhost:5000/api/process_image', json=payload, timeout=300)
        status_indicator.text(f"API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            status_indicator.success("API request successful!")
            return response.json()
        else:
            status_indicator.error(f"API Error: {response.status_code}")
            st.error(f"API Response: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        status_indicator.error("❌ Connection Error: Cannot connect to the backend server")
        st.error("Make sure the Flask backend is running on http://localhost:5000")
        return None
    except requests.exceptions.Timeout:
        status_indicator.error("❌ Request Timeout: The backend server took too long to respond")
        return None
    except Exception as e:
        status_indicator.error(f"❌ Error connecting to API: {str(e)}")
        return None

# Initialize session state for progress tracking
if 'processing_state' not in st.session_state:
    st.session_state.processing_state = 'idle'

# Check API connection at startup
with st.sidebar:
    st.header("API Connection")
    if st.button("Test Backend Connection"):
        test_api_connection()

# Main UI
st.header("Upload an image")
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    try:
        # Original image display
        original_img = Image.open(uploaded_image)
        original_img = original_img.resize((256, 256))
        
        st.image(original_img, caption="Input Image", use_column_width=False, width=256)
        
        # Display image details
        img_format = original_img.format
        img_size = original_img.size
        st.text(f"Image format: {img_format}, Size: {img_size[0]}x{img_size[1]}")
        
        # Process button
        if st.button("Process Image") or st.session_state.processing_state == 'processing':
            if st.session_state.processing_state == 'idle':
                st.session_state.processing_state = 'processing'
                st.rerun()
            
            # Set up progress interface
            st.header("Processing Steps")
            
            # Status indicators
            with st.container():
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.subheader("1. Image Loaded")
                    st.image(original_img, use_column_width=False, width=256)
                    step1_status = st.success("Completed")
                
                with col2:
                    st.subheader("2. Face Detection")
                    face_placeholder = st.empty()
                    class_placeholder = st.empty()
                    step2_status = st.info("Processing...")
                
                with col3:
                    st.subheader("3. Mask Generation")
                    mask_placeholder = st.empty()
                    step3_status = st.info("Processing...")
                
                with col4:
                    st.subheader("4. Image Generation")
                    gen_placeholder = st.empty()
                    step4_status = st.info("Processing...")
            
            # Results placeholder
            result_container = st.container()
            
            # Process the image
            with st.spinner("Processing image with backend... This may take a minute."):
                result = process_image(original_img)
                
            if result:
                # Update face detection status
                step2_status.success("Completed")
                class_name = result.get('class_name', 'Unknown Class')
                face_placeholder.write("Face Detected")
                class_placeholder.info(f"Class: {class_name}")
                
                # Update mask generation status
                step3_status.success("Completed")
                mask_image = Image.open(io.BytesIO(base64.b64decode(result['mask_image'])))
                mask_placeholder.image(mask_image, caption="Binary Mask", use_column_width=False, width=256)
                
                # Update generation status
                step4_status.success("Completed")
                
                # Convert the base64 images back to displayable format
                combined_image = Image.open(io.BytesIO(base64.b64decode(result['combined_image'])))
                generated_image = Image.open(io.BytesIO(base64.b64decode(result['generated_image'])))
                
                # Resize to 256x256
                combined_image = combined_image.resize((256, 256))
                generated_image = generated_image.resize((256, 256))
                
                gen_placeholder.image(generated_image, caption="Generated", use_column_width=False, width=256)
                
                # Final results
                with result_container:
                    st.header("Final Results")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.image(original_img, caption="Original Image", use_column_width=False, width=256)
                    
                    with col2:
                        st.markdown("<div style='text-align: center; font-size: 24px;'>➡️</div>", unsafe_allow_html=True)
                    
                    with col3:
                        st.image(generated_image, caption="Generated Image", use_column_width=False, width=256)
                    
                    with col4:
                        st.markdown("<div style='text-align: center; font-size: 24px;'>=</div>", unsafe_allow_html=True)
                    
                    with col5:
                        st.image(combined_image, caption="Combined Result", use_column_width=False, width=256)
                    
                    # Download buttons
                    st.subheader("Download Results")
                    download_col1, download_col2 = st.columns(2)
                    
                    with download_col1:
                        # Save generated image to bytes
                        buf = io.BytesIO()
                        generated_image.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        st.download_button(
                            label="Download Generated Image",
                            data=byte_im,
                            file_name="generated_image.png",
                            mime="image/png"
                        )
                    
                    with download_col2:
                        # Save combined image to bytes
                        buf = io.BytesIO()
                        combined_image.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        st.download_button(
                            label="Download Combined Image",
                            data=byte_im,
                            file_name="combined_image.png",
                            mime="image/png"
                        )
                
                # Reset processing state
                st.session_state.processing_state = 'idle'
            else:
                st.error("Failed to process the image. Please try again.")
                st.session_state.processing_state = 'idle'
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.session_state.processing_state = 'idle'
else:
    # Instructions when no image is uploaded
    st.info("Please upload an image to begin processing.")
    
# Add some information about the app
with st.expander("About this app"):
    st.write("""
    This application uses advanced AI models to detect and restore face images.
    The process involves:
    1. Face detection and classification
    2. Binary mask generation using a segmentation model 
    3. Image generation with a GAN model
    4. Combining the generated image with the original for the final result
    """)

# Debug section in sidebar
with st.sidebar:
    st.header("Troubleshooting")
    if st.checkbox("Show debug info"):
        st.subheader("Session State")
        st.write(st.session_state)