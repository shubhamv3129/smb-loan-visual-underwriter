"""
Unit tests for utils.py module functions.
Validates coordinate conversion, EXIF parsing, Gemini interactions, Secret Manager access,
and Google Maps client configuration with mocked external API calls.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import utils


def test_decimal_coords_cardinal_directions():
    """Verify DMS to decimal conversion for all four cardinal directions."""
    # North: positive
    assert pytest.approx(utils.decimal_coords((12, 30, 0), "N"), rel=1e-5) == 12.5
    # South: negative
    assert pytest.approx(utils.decimal_coords((12, 30, 0), "S"), rel=1e-5) == -12.5
    # East: positive
    assert pytest.approx(utils.decimal_coords((77, 15, 36), "E"), rel=1e-5) == 77.26
    # West: negative
    assert pytest.approx(utils.decimal_coords((77, 15, 36), "W"), rel=1e-5) == -77.26


def test_initialize_gemini_model_custom_project():
    """Verify Gemini model initialization when custom project ID is provided."""
    with patch.object(utils, "PROJECT_ID", "my-custom-project"), \
         patch.object(utils, "LOCATION", "us-central1"), \
         patch.object(utils.vertexai, "init") as mock_init, \
         patch.object(utils, "GenerativeModel") as mock_model_cls:
        
        mock_instance = MagicMock()
        mock_model_cls.return_value = mock_instance
        
        result = utils.initialize_gemini_model("gemini-1.5-flash")
        
        mock_init.assert_called_once_with(project="my-custom-project", location="us-central1")
        mock_model_cls.assert_called_once_with("gemini-1.5-flash")
        assert result == mock_instance


def test_initialize_gemini_model_default_placeholder_project():
    """Verify vertexai.init is skipped when default placeholder project ID is used."""
    with patch.object(utils, "PROJECT_ID", "your-gcp-project-id"), \
         patch.object(utils.vertexai, "init") as mock_init, \
         patch.object(utils, "GenerativeModel") as mock_model_cls:
        
        utils.initialize_gemini_model()
        mock_init.assert_not_called()
        mock_model_cls.assert_called_once()


def test_extract_image_metadata_valid_exif():
    """Verify extraction of GPS coordinates and timestamp from valid EXIF data."""
    mock_img = MagicMock()
    mock_img.has_exif = True
    mock_img.gps_latitude = (12, 30, 0)
    mock_img.gps_latitude_ref = "N"
    mock_img.gps_longitude = (77, 30, 0)
    mock_img.gps_longitude_ref = "E"
    mock_img.datetime_original = "2024:06:01 10:15:30"

    with patch("builtins.open", MagicMock()), \
         patch.object(utils, "ExifImage", return_value=mock_img):
        
        metadata = utils.extract_image_metadata("dummy_shop.jpg")
        
        assert metadata is not None
        assert metadata["imageTakenTime"] == "2024:06:01 10:15:30"
        assert pytest.approx(metadata["geolocation_lat"], rel=1e-5) == 12.5
        assert pytest.approx(metadata["geolocation_lng"], rel=1e-5) == 77.5


def test_extract_image_metadata_no_exif():
    """Verify None is returned when image has no EXIF data."""
    mock_img = MagicMock()
    mock_img.has_exif = False

    with patch("builtins.open", MagicMock()), \
         patch.object(utils, "ExifImage", return_value=mock_img):
        
        metadata = utils.extract_image_metadata("dummy_no_exif.jpg")
        assert metadata is None


def test_extract_image_metadata_missing_gps_attributes():
    """Verify None is returned when EXIF exists but lacks GPS attributes."""
    mock_img = MagicMock()
    mock_img.has_exif = True
    del mock_img.gps_latitude  # triggers AttributeError

    with patch("builtins.open", MagicMock()), \
         patch.object(utils, "ExifImage", return_value=mock_img):
        
        metadata = utils.extract_image_metadata("dummy_no_gps.jpg")
        assert metadata is None


def test_extract_image_metadata_io_exception():
    """Verify None is returned and exception handled when image file cannot be opened."""
    with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
        metadata = utils.extract_image_metadata("missing_file.jpg")
        assert metadata is None


def test_generate_gemini_response():
    """Verify generate_gemini_response formats parts and returns response text."""
    mock_model = MagicMock()
    mock_response = MagicMock(text="Evaluated Credit Risk: LOW")
    mock_model.generate_content.return_value = mock_response

    parts = ["image_part_1", "image_part_2"]
    prompt = "Analyze the inventory value"
    
    result = utils.generate_gemini_response(mock_model, parts, prompt, "Processing...")
    
    mock_model.generate_content.assert_called_once_with(["image_part_1", "image_part_2", prompt])
    assert result == "Evaluated Credit Risk: LOW"


def test_generate_gemini_response_with_grounding():
    """Verify generate_gemini_response_with_grounding attaches Search Grounding tool."""
    mock_model = MagicMock()
    mock_response = MagicMock(text="Grounded appraisal: INR 2,50,000")
    mock_model.generate_content.return_value = mock_response

    with patch.object(utils.Tool, "from_google_search_retrieval", return_value="mock_search_tool") as mock_tool_fn:
        result = utils.generate_gemini_response_with_grounding(
            mock_model, [], "Appraise inventory in Bengaluru", "Searching..."
        )
        mock_tool_fn.assert_called_once()
        mock_model.generate_content.assert_called_once_with("Appraise inventory in Bengaluru", tools=["mock_search_tool"])
        assert result == "Grounded appraisal: INR 2,50,000"


def test_load_image_success():
    """Verify load_image loads image from file and creates Part object."""
    mock_loaded_img = MagicMock()
    mock_part = MagicMock()
    
    with patch.object(utils.Image, "load_from_file", return_value=mock_loaded_img) as mock_load, \
         patch.object(utils.Part, "from_image", return_value=mock_part) as mock_from_img:
        
        part = utils.load_image("shop_front.jpg")
        mock_load.assert_called_once_with("shop_front.jpg")
        mock_from_img.assert_called_once_with(mock_loaded_img)
        assert part == mock_part


def test_load_image_failure():
    """Verify load_image returns None when loading fails."""
    with patch.object(utils.Image, "load_from_file", side_effect=IOError("Corrupt image")):
        part = utils.load_image("corrupt.jpg")
        assert part is None


def test_create_text_part():
    """Verify create_text_part forwards content to Part.from_text."""
    with patch.object(utils.Part, "from_text", return_value="mock_text_part") as mock_from_text:
        part = utils.create_text_part("Sample text prompt")
        mock_from_text.assert_called_once_with("Sample text prompt")
        assert part == "mock_text_part"


def test_create_audio_part():
    """Verify create_audio_part forwards URI and mime_type to Part.from_uri."""
    with patch.object(utils.Part, "from_uri", return_value="mock_audio_part") as mock_from_uri:
        part = utils.create_audio_part("gs://bucket/sample.mp3")
        mock_from_uri.assert_called_once_with("gs://bucket/sample.mp3", mime_type="audio/mp3")
        assert part == "mock_audio_part"


def test_access_secret_version_success():
    """Verify Secret Manager payload retrieval and UTF-8 decoding."""
    mock_client = MagicMock()
    mock_payload = MagicMock(data=b"super_secret_maps_key_123")
    mock_response = MagicMock(payload=mock_payload)
    mock_client.access_secret_version.return_value = mock_response

    with patch.object(utils.secretmanager, "SecretManagerServiceClient", return_value=mock_client):
        val = utils.access_secret_version("GoogleMapsAPIKey", "test-project")
        assert val == "super_secret_maps_key_123"
        mock_client.access_secret_version.assert_called_once_with(
            request={"name": "projects/test-project/secrets/GoogleMapsAPIKey/versions/latest"}
        )


def test_access_secret_version_error():
    """Verify access_secret_version returns None on Secret Manager exception."""
    mock_client = MagicMock()
    mock_client.access_secret_version.side_effect = Exception("PermissionDenied")

    with patch.object(utils.secretmanager, "SecretManagerServiceClient", return_value=mock_client):
        val = utils.access_secret_version("NonExistentSecret")
        assert val is None


def test_get_google_maps_client_from_env():
    """Verify Google Maps client initialization using environment variable."""
    with patch.dict(os.environ, {"GOOGLE_MAPS_API_KEY": "AIzaSyTestEnvKey"}), \
         patch.object(utils.googlemaps, "Client") as mock_maps_client_cls:
        
        mock_instance = MagicMock()
        mock_maps_client_cls.return_value = mock_instance
        
        client = utils.get_google_maps_client()
        mock_maps_client_cls.assert_called_once_with(key="AIzaSyTestEnvKey")
        assert client == mock_instance


def test_get_google_maps_client_from_secret_manager():
    """Verify Google Maps client initialization fallback to Secret Manager."""
    env_clean = {k: v for k, v in os.environ.items() if k != "GOOGLE_MAPS_API_KEY"}
    with patch.dict(os.environ, env_clean, clear=True), \
         patch.object(utils, "access_secret_version", return_value="SecretManagerMapsKey") as mock_sec, \
         patch.object(utils.googlemaps, "Client") as mock_maps_client_cls:
        
        mock_instance = MagicMock()
        mock_maps_client_cls.return_value = mock_instance
        
        client = utils.get_google_maps_client()
        mock_sec.assert_called_once_with("GoogleMapsAPIKey")
        mock_maps_client_cls.assert_called_once_with(key="SecretManagerMapsKey")
        assert client == mock_instance


def test_get_google_maps_client_missing_key():
    """Verify get_google_maps_client returns None when no key is found."""
    env_clean = {k: v for k, v in os.environ.items() if k != "GOOGLE_MAPS_API_KEY"}
    with patch.dict(os.environ, env_clean, clear=True), \
         patch.object(utils, "access_secret_version", return_value=None):
        
        client = utils.get_google_maps_client()
        assert client is None


def test_read_text_file_success():
    """Verify read_text_file successfully reads content from disk."""
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
        tmp.write("Loan applicant documentation test")
        tmp_path = tmp.name

    try:
        content = utils.read_text_file(tmp_path)
        assert content == "Loan applicant documentation test"
    finally:
        os.remove(tmp_path)


def test_read_text_file_missing():
    """Verify read_text_file returns None when file does not exist."""
    content = utils.read_text_file("/path/to/definitely/non_existent_file.txt")
    assert content is None
