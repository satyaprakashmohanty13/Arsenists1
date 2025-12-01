import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sudoku_vision import SudokuVision
from sudoku_solver import SudokuSolver
import pandas as pd

st.set_page_config(page_title="Sudoku Solver", layout="wide")

st.title("AI Sudoku Solver")
st.write("Upload a Sudoku image, and the AI will solve it for you!")

# Initialize vision model (cached to avoid reloading)
@st.cache_resource
def load_vision_model():
    return SudokuVision()

try:
    vision = load_vision_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to opencv image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Handle RGBA images
    if img_array.shape[-1] == 4:
         img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:
        # Convert RGB to BGR (opencv uses BGR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption='Uploaded Image', width="stretch")

    with st.spinner('Processing image...'):
        try:
            # Recognize grid
            recognized_grid, display_grid_img = vision.recognize_grid_from_image(img_array)

            # Convert BGR to RGB for display
            display_grid_img_rgb = cv2.cvtColor(display_grid_img, cv2.COLOR_BGR2RGB)

            with col2:
                st.image(display_grid_img_rgb, caption='Processed Image', width="stretch")

            st.subheader("Recognized Grid")
            st.write("Please verify the numbers below and edit if necessary.")

            # Create a DataFrame for editing
            df = pd.DataFrame(recognized_grid)
            # Use data editor
            edited_df = st.data_editor(df, height=350, use_container_width=True)

            if st.button("Solve Sudoku"):
                # Convert back to list of lists
                final_grid = edited_df.values.tolist()

                # Convert to string format expected by solver
                sudoku_str = ""
                for i in range(9):
                    if i > 0:
                        sudoku_str += "/"
                    for j in range(9):
                        sudoku_str += str(final_grid[i][j])

                # Solve
                solver = SudokuSolver(sudoku_str)
                if solver.solve():
                    st.success("Sudoku Solved!")

                    # Display solution
                    solution_grid = solver.grid
                    st.subheader("Solution:")

                    # Create a nice display for the solution
                    solution_df = pd.DataFrame(solution_grid)
                    st.dataframe(solution_df, use_container_width=True)
                else:
                    st.error("Unable to solve this Sudoku. Please check the numbers.")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
