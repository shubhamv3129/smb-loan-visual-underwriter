"""
Utility functions for the AI Demo Loan project.
Handles API clients, image processing, and Gemini model interactions.
"""
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from exif import Image as ExifImage
from google.cloud import secretmanager
import googlemaps
import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, Image, Part, Tool, grounding

# Load environment variables from .env if present
load_dotenv()

# Constants loaded from environment with sensible defaults
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")


def initialize_gemini_model(model_name: str = MODEL_NAME) -> GenerativeModel:
    """
    Initializes Vertex AI and returns a GenerativeModel instance.

    Args:
        model_name (str): The name of the Gemini model to initialize.

    Returns:
        GenerativeModel: The initialized model.
    """
    if PROJECT_ID and PROJECT_ID != "your-gcp-project-id":
        vertexai.init(project=PROJECT_ID, location=LOCATION)
    return GenerativeModel(model_name)


def decimal_coords(coords: Tuple[float, float, float], ref: str) -> float:
    """
    Converts GPS coordinates from degrees/minutes/seconds to decimal degrees.

    Args:
        coords (tuple): Coordinates in (degrees, minutes, seconds).
        ref (str): The reference direction ('N', 'S', 'E', 'W').

    Returns:
        float: The decimal representation of the coordinates.
    """
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref in ("S", "W"):
        decimal_degrees = -decimal_degrees
    return decimal_degrees


def extract_image_metadata(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Extracts GPS coordinates and timestamp from an image's EXIF data.

    Args:
        image_path (str): The path to the image file.

    Returns:
        dict: A dictionary containing the image taken time, latitude, and longitude.
              Returns None if metadata cannot be extracted.
    """
    try:
        with open(image_path, "rb") as src:
            img = ExifImage(src)
        if img.has_exif:
            try:
                coords = (
                    decimal_coords(img.gps_latitude, img.gps_latitude_ref),
                    decimal_coords(img.gps_longitude, img.gps_longitude_ref),
                )
                return {
                    "imageTakenTime": getattr(img, "datetime_original", "Unknown"),
                    "geolocation_lat": coords[0],
                    "geolocation_lng": coords[1],
                }
            except AttributeError:
                print("No Coordinates found in EXIF.")
                return None
        else:
            st.write("The image has no EXIF data.")
            return None
    except Exception as exc:
        print(f"Error processing image file: {exc}")
        return None


def generate_gemini_response(
    model: GenerativeModel,
    parts: List[Any],
    prompt: str,
    spinner_message: str,
) -> str:
    """
    Generates content using the Gemini model with a Streamlit spinner.

    Args:
        model (GenerativeModel): The Gemini model instance.
        parts (list): A list of media parts (e.g., images).
        prompt (str): The text prompt.
        spinner_message (str): The message to display while loading.

    Returns:
        str: The generated text response.
    """
    with st.spinner(spinner_message):
        response = model.generate_content(parts + [prompt])
    return response.text


def generate_gemini_response_with_grounding(
    model: GenerativeModel,
    parts: List[Any],
    prompt: str,
    spinner_message: str,
) -> str:
    """
    Generates content using the Gemini model with Google Search Grounding.

    Args:
        model (GenerativeModel): The Gemini model instance.
        parts (list): A list of media parts (kept for signature consistency).
        prompt (str): The text prompt.
        spinner_message (str): The message to display while loading.

    Returns:
        str: The generated text response.
    """
    tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
    with st.spinner(spinner_message):
        response = model.generate_content(prompt, tools=[tool])
    return response.text


def load_image(image_path: str) -> Optional[Part]:
    """
    Loads an image from the filesystem and returns a Gemini Part object.

    Args:
        image_path (str): The path to the image.

    Returns:
        Part: The Gemini Part object, or None if loading fails.
    """
    try:
        return Part.from_image(Image.load_from_file(image_path))
    except Exception as exc:
        st.error(f"Error loading image: {exc}")
        return None


def create_text_part(text_content: str) -> Part:
    """
    Creates a text Part object for Gemini.

    Args:
        text_content (str): The text content.

    Returns:
        Part: The generated Part object.
    """
    return Part.from_text(text_content)


def create_audio_part(audio_uri: str) -> Part:
    """
    Creates an audio Part object from a URI for Gemini.

    Args:
        audio_uri (str): The URI of the audio file.

    Returns:
        Part: The generated Part object.
    """
    return Part.from_uri(audio_uri, mime_type="audio/mp3")


def access_secret_version(secret_id: str, project_id: str = PROJECT_ID) -> Optional[str]:
    """
    Accesses a secret payload from Google Cloud Secret Manager.

    Args:
        secret_id (str): The ID of the secret.
        project_id (str): The Google Cloud project ID.

    Returns:
        str: The decoded secret payload, or None if an error occurs.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    try:
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as exc:
        print(f"Error accessing secret {secret_id}: {exc}")
        return None


def get_google_maps_client() -> Optional[googlemaps.Client]:
    """
    Retrieves the Google Maps API key from environment or Secret Manager
    and initializes a Google Maps client.

    Returns:
        googlemaps.Client: The initialized client, or None if API key retrieval fails.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY") or access_secret_version("GoogleMapsAPIKey")
    if not api_key:
        st.error(
            "Could not retrieve Google Maps API key. Set GOOGLE_MAPS_API_KEY in .env "
            "or configure 'GoogleMapsAPIKey' in Google Cloud Secret Manager."
        )
        return None
    return googlemaps.Client(key=api_key)


def read_text_file(file_path: str) -> Optional[str]:
    """
    Reads and returns the content of a text file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the file, or None if an error occurs.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as exc:
        st.error(f"Error reading the file: {exc}")
        return None
