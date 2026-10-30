"""
Unit tests for feature_shop_property.py.
Validates the end-to-end shop insight pipeline, OCR processing, Google Maps integration,
and Gemini grounding appraisal with mocked external API calls.
"""

from unittest.mock import MagicMock, patch

import pytest
import feature_shop_property


def test_process_shop_image_insights_complete_flow():
    """Verify complete shop property processing pipeline with mocked APIs."""
    mock_model = MagicMock()
    mock_image_part = MagicMock()

    # Mock load_image returning valid parts for all images
    mock_load_image = MagicMock(return_value=mock_image_part)
    
    # Mock Gemini responses: 1st for OCR shop name, 2nd for inventory itemization
    mock_gemini_response = MagicMock(side_effect=[
        "Shree Ganesh Kirana Store",
        "1, Toor Dal, 50kg\n2, Basmati Rice, 100kg\n3, Sunflower Oil, 20L"
    ])
    
    # Mock image metadata
    mock_metadata = {
        "geolocation_lat": 12.9716,
        "geolocation_lng": 77.5946,
        "imageTakenTime": "2024:05:10 14:22:00"
    }
    
    # Mock Google Maps client
    mock_maps_client = MagicMock()
    mock_maps_client.places_nearby.side_effect = [
        {"results": [{"name": "Shree Ganesh Kirana Store", "vicinity": "MG Road, Bengaluru", "rating": 4.6}]},
        {"results": [{"name": "Reliance Fresh", "vicinity": "Brigade Road", "rating": 4.2}]}
    ]

    mock_grounded_response = MagicMock(return_value="Total value: INR 1,50,000. Low credit risk.")

    with patch.object(feature_shop_property, "load_image", mock_load_image), \
         patch.object(feature_shop_property, "generate_gemini_response", mock_gemini_response), \
         patch.object(feature_shop_property, "extract_image_metadata", return_value=mock_metadata), \
         patch.object(feature_shop_property, "get_google_maps_client", return_value=mock_maps_client), \
         patch.object(feature_shop_property, "generate_gemini_response_with_grounding", mock_grounded_response), \
         patch.object(feature_shop_property, "initialize_gemini_model", return_value=mock_model):

        feature_shop_property.process_shop_image_insights(
            "shop_front.jpg", "shop_inventory_1.jpg", "shop_inventory_2.jpg", mock_model
        )

        # Verify Google Maps was called with coordinates and OCR keyword
        mock_maps_client.places_nearby.assert_any_call(
            location=(12.9716, 77.5946),
            radius=100,
            keyword="Shree Ganesh Kirana Store"
        )
        mock_grounded_response.assert_called_once()


def test_process_shop_image_insights_missing_front_image():
    """Verify early exit when storefront image fails to load."""
    mock_model = MagicMock()
    with patch.object(feature_shop_property, "load_image", return_value=None):
        result = feature_shop_property.process_shop_image_insights(
            "missing_front.jpg", "inv1.jpg", "inv2.jpg", mock_model
        )
        assert result is None


def test_process_shop_image_insights_ocr_fails():
    """Verify early exit when OCR returns empty or None."""
    mock_model = MagicMock()
    with patch.object(feature_shop_property, "load_image", return_value=MagicMock()), \
         patch.object(feature_shop_property, "generate_gemini_response", return_value=None):
        
        result = feature_shop_property.process_shop_image_insights(
            "front.jpg", "inv1.jpg", "inv2.jpg", mock_model
        )
        assert result is None


def test_process_shop_image_insights_missing_inventory_images():
    """Verify early exit when inventory images fail to load after successful OCR."""
    mock_model = MagicMock()
    # Front image succeeds, inventory images fail
    mock_load_image = MagicMock(side_effect=[MagicMock(), None, None])

    with patch.object(feature_shop_property, "load_image", mock_load_image), \
         patch.object(feature_shop_property, "generate_gemini_response", return_value="Test Store"), \
         patch.object(feature_shop_property, "extract_image_metadata", return_value=None), \
         patch.object(feature_shop_property, "get_google_maps_client", return_value=None):

        result = feature_shop_property.process_shop_image_insights(
            "front.jpg", "inv1.jpg", "inv2.jpg", mock_model
        )
        assert result is None


def test_process_shop_image_insights_maps_api_error():
    """Verify graceful handling when Google Maps API throws an ApiError."""
    import googlemaps.exceptions
    mock_model = MagicMock()
    mock_load_image = MagicMock(return_value=MagicMock())
    mock_gemini = MagicMock(side_effect=["Supermarket ABC", "1, Milk, 10 packets"])
    
    mock_maps_client = MagicMock()
    mock_maps_client.places_nearby.side_effect = googlemaps.exceptions.ApiError("OVER_QUERY_LIMIT")

    mock_grounded = MagicMock(return_value="Risk evaluation completed.")

    with patch.object(feature_shop_property, "load_image", mock_load_image), \
         patch.object(feature_shop_property, "generate_gemini_response", mock_gemini), \
         patch.object(feature_shop_property, "extract_image_metadata", return_value={"geolocation_lat": 10.0, "geolocation_lng": 20.0}), \
         patch.object(feature_shop_property, "get_google_maps_client", return_value=mock_maps_client), \
         patch.object(feature_shop_property, "generate_gemini_response_with_grounding", mock_grounded), \
         patch.object(feature_shop_property, "initialize_gemini_model", return_value=mock_model):

        # Should handle error gracefully without raising
        feature_shop_property.process_shop_image_insights(
            "front.jpg", "inv1.jpg", "inv2.jpg", mock_model
        )
        mock_grounded.assert_called_once()


def test_shop_property_section_render():
    """Verify shop_property_section renders UI components properly."""
    import streamlit as st
    mock_model = MagicMock()
    st.button = MagicMock(return_value=False)

    feature_shop_property.shop_property_section(mock_model)
    st.markdown.assert_called()


def test_shop_property_section_trigger_analysis():
    """Verify shop_property_section triggers processing when button is clicked."""
    import streamlit as st
    mock_model = MagicMock()
    st.button = MagicMock(return_value=True)

    with patch.object(feature_shop_property, "process_shop_image_insights") as mock_process:
        feature_shop_property.shop_property_section(mock_model)
        mock_process.assert_called_once_with(
            "shop_front.jpg", "shop_inventory_1.jpg", "shop_inventory_2.jpg", mock_model
        )
