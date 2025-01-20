import streamlit as st

# Title of the app
st.title("Simple Streamlit App")

# Text input
user_input = st.text_input("Enter some text:")

# Number input
number = st.number_input("Enter a number:", min_value=0, max_value=100, value=0, step=1)

# Checkbox
if st.checkbox("Show Message"):
    st.write("Hello, Streamlit user!")

# Button to display input
if st.button("Submit"):
    st.write(f"You entered: {user_input}")
    st.write(f"Your number is: {number}")

# Select box
option = st.selectbox(
    "Select a color:",
    ["Red", "Green", "Blue"]
)

st.write(f"You selected: {option}")

# Display image from URL
st.image("https://dummyimage.com/150", caption="Sample Image", use_container_width=True)

# Footer
st.markdown(
    """
    ---
    Simple Streamlit App Example
    """
)
