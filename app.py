import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from model import SimpleNeuralNetwork

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("🔢 MNIST Neural Network from Scratch")
st.write("Draw a digit (0-9) in the box below and let the model predict it!")

@st.cache_resource
def load_model():
    model = SimpleNeuralNetwork()
    model.load_weights()
    return model
model = load_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Draw Here")
    canvas_result = st_canvas(
        stroke_color="#FFFFFF",
        stroke_width=20,
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True,
        return_image_data=True,
    )

with col2:
    st.subheader("Prediction")
    if (
        canvas_result.image_data is not None and np.sum(canvas_result.image_data[:, :, :3])
    ):
        # 1. Convert RGBA to Grayscale
        img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert(
            "L"
        )

        # 2. Get bounding box of non-zero pixels (the drawn stroke)
        bbox = img.getbbox()

        if bbox:
            # Crop tightly to the digit
            cropped = img.crop(bbox)

            # Create a square canvas matching the largest dimension + padding
            width, height = cropped.size
            max_dim = max(width, height)
            padding = int(max_dim * 0.4)  # Add 20% margin on each side
            square_size = max_dim + padding

            # Paste cropped digit into center of square black image
            centered_img = Image.new("L", (square_size, square_size), 0)
            paste_x = (square_size - width) // 2
            paste_y = (square_size - height) // 2
            centered_img.paste(cropped, (paste_x, paste_y))

            # 3. Resize centered digit to 28x28 (MNIST standard)
            img_resized = centered_img.resize((28, 28), Image.Resampling.LANCZOS)
        else:
            img_resized = img.resize((28, 28))
            

        # 4. Normalize to [0.0, 1.0] and flatten to (1, 784)
        img_array = np.array(img_resized) / 255.0
        input_vector = img_array.reshape(1, 784)


        probabilities = model.Forward(input_vector)[0]
        predicted_digit = np.argmax(probabilities)

        st.metric(label="Predicted Digit", value=f"{predicted_digit}")
        st.write(f"Confidence: {probabilities[predicted_digit]*100:.2f}")
        st.bar_chart(probabilities)

    else:
        st.write("Draw a digit in tho box to see predictions!")

