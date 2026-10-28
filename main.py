"""
Main entry point for the AI Demo Loan application.
Demonstrates the usage of the shop property credit risk evaluation feature.
"""
import streamlit as st
from feature_shop_property import shop_property_section
from utils import initialize_gemini_model


def example_shop_property() -> None:
    """
    Example function to showcase the shop property feature.
    Initializes the model and sets up the Streamlit UI.
    """
    st.set_page_config(page_title="AI Demo Loan", page_icon="🏦", layout="wide")
    st.title("🏦 AI Demo Loan Application")
    st.write(
        "Welcome to the AI Loan Underwriting Demo. This tool analyzes storefront "
        "and inventory images to evaluate SMB credit risk."
    )

    # Initialize the required Gemini model
    model = initialize_gemini_model()

    # Manage UI state if needed for multiple pages
    if (
        "selected_option" not in st.session_state
        or st.session_state.selected_option == "shop_property"
    ):
        shop_property_section(model)


def main() -> None:
    """
    Main entry point to trigger the application workflow.
    """
    example_shop_property()


if __name__ == "__main__":
    main()
