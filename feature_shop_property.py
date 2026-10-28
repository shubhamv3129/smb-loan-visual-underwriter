"""
Feature module containing the logic for processing shop properties.
Handles OCR, Google Maps lookups, and inventory analysis using Gemini.
"""
import streamlit as st
import googlemaps
from utils import (
    load_image,
    generate_gemini_response,
    generate_gemini_response_with_grounding,
    extract_image_metadata,
    get_google_maps_client,
    initialize_gemini_model
)


def process_shop_image_insights(shop_front_image, shop_inventory_1, shop_inventory_2, model):
    """
    Processes shop images to extract insights, verify locations, and appraise inventory.

    Args:
        shop_front_image (str): Path to the shop front image.
        shop_inventory_1 (str): Path to the first inventory image.
        shop_inventory_2 (str): Path to the second inventory image.
        model (GenerativeModel): The initialized Gemini model.

    Returns:
        None
    """
    # Initialize place to prevent UnboundLocalError
    place = {}

    st.markdown("##### Examining Store Front Image")
    image_part = load_image(shop_front_image)
    if not image_part:
        return None

    # Prompt for extracting shop name using OCR
    shop_front_ocr_prompt = """
    You are an expert OCR Model.
    Detect the name of the establishment shown in this picture. Only English output.
    Print only the name of the establishment, no other text or explanation needed.
    """
    
    # Analyze the front image for the shop name
    shop_name = generate_gemini_response(
        model, [image_part], shop_front_ocr_prompt, "Working on it!"
    )
    if not shop_name:
        return None

    st.write("Shop's name detected is: " + shop_name)

    # Extract geolocation metadata from the image
    image_lat_long = extract_image_metadata(shop_front_image)

    if image_lat_long:
        image_lat = image_lat_long['geolocation_lat']
        image_lng = image_lat_long['geolocation_lng']
    else:
        image_lat, image_lng = None, None

    st.markdown("##### Looking up Google Places with this name...")
    api_client = get_google_maps_client()

    if api_client and image_lat and image_lng:
        try:
            # Search for the specific shop name nearby
            places_result = api_client.places_nearby(
                location=(image_lat, image_lng),
                radius=100,
                keyword=shop_name
            )
            tmp = places_result.get('results', [])

            if tmp:
                place = tmp[0]

            st.write("###### Shop details from Google Maps:")
            for p in tmp:
                st.write(f"- {p['name']}")
                st.write(f"  Address: {p.get('vicinity', 'Address not available')}")
                st.write(f"  Rating: {p.get('rating', 'Rating not available')}")

            # Search for similar shops nearby
            nearby_places_result = api_client.places_nearby(
                location=(image_lat, image_lng),
                radius=500,
                type='grocery_or_supermarket'
            )
            nearby_tmp = nearby_places_result.get('results', [])
            st.write("###### Similar shops found nearby, from Google Maps:")
            for nearby_place in nearby_tmp:
                st.markdown(f"  ###### Name: {nearby_place['name']}")
                st.write(f"  Address: {nearby_place.get('vicinity', 'Address not available')}")
                st.write(f"  Rating: {nearby_place.get('rating', 'Rating not available')}")

        except googlemaps.exceptions.ApiError as e:
            print(f"Error accessing Google Maps API: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    else:
        st.write("Skipping Google Maps lookup due to missing coordinates or API client.")

    st.markdown("##### Now examining the store's inventory from the images...")
    
    # Load inventory images
    inventory_image_1 = load_image(shop_inventory_1)
    inventory_image_2 = load_image(shop_inventory_2)
    if not inventory_image_1 or not inventory_image_2:
        return None

    # Prompt for analyzing inventory
    inventory_analysis_prompt = """
    You are an expert Grocery Store Inventory Checker, capable of noticing the smallest and minutest details in a shop.
    Collect information about the brands, types of goods, quantity seen in this picture.
    Output the values in a comma separated fashion with columns as follows - Sr no, Goods Description, Quantity - each type of item you see should be reported as a row within these columns.
    Try to cover as many types of goods as possible.
    """

    # Extract inventory details
    inventory_items = generate_gemini_response(
        model, 
        [inventory_image_1, inventory_image_2], 
        inventory_analysis_prompt, 
        "Analyzing store inventory..."
    )
    if not inventory_items:
        return None

    # Prompt for grounding and appraisal
    grounding_prompt = f"""
    You are an expert Goods Price Appraiser, working for the largest Private Sector Bank in India.
    For each line in the input, perform a Google Search to find out the approximate price of these goods.
    Give me all line items, with their total prices - Quantity multipled by approximate prices in INR.
    At the end, give me one single number called Total approximate value of goods.
    The store is located in {place.get('vicinity', 'an unknown location')}
    All currencies to be in INR.
    From a Loan Perspective, given what you know from the location, goods present - what is your recommendation on Credit Risk, if this shop owner were to apply for a loan?
    Input for you: {inventory_items}
    """

    # Use the appropriate model for grounding
    grounding_model = initialize_gemini_model("gemini-2.5-pro")

    validation_response = generate_gemini_response_with_grounding(
        grounding_model,
        [], 
        grounding_prompt,
        "Analyzing with grounding"
    )
    
    st.markdown("##### Inventory Analysis Report")
    st.write(validation_response)


def shop_property_section(model):
    """
    Renders the UI for the shop property analysis and handles user interaction.

    Args:
        model (GenerativeModel): The initialized Gemini model.
    """
    st.markdown("#### OCR using Gemini, Verification of Shop Address using Google Maps, and Calculation of Inventory Value using Gemini and Google Search grounding")
    
    shop_front_image = "shop_front.jpg"
    shop_inventory_1 = "shop_inventory_1.jpg"
    shop_inventory_2 = "shop_inventory_2.jpg"

    # Display images in three columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(shop_front_image, caption="Shop Front", use_container_width=True)
    with col2:
        st.image(shop_inventory_1, caption="Shop Inventory", use_container_width=True)
    with col3:
        st.image(shop_inventory_2, caption="Shop Inventory", use_container_width=True)
        
    # Button to trigger analysis
    if st.button("Generate Insights", key='home_image_button', type="primary"):
        process_shop_image_insights(shop_front_image, shop_inventory_1, shop_inventory_2, model)
