// File: /lib/api.js

// This is the URL where your Python backend is running.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Analyzes media by sending it to the new v3 backend's video processing endpoint.
 *
 * This function is designed to work with server_v3.py.
 * It sends a single video file, and the backend handles all
 * (Phase 1, Phase 2, and Audio) extraction and analysis.
 *
 * @param {File} videoFile - The video file. This is REQUIRED.
 * @param {object} options - The context options (e.g., zone_type, is_night_time, sample_fps).
 * @returns {Promise<object>} - The VideoAnalysisResponse from the backend.
 */
export const analyzeMedia = async (videoFile, options = {}) => {
  
  if (!videoFile) {
    throw new Error("A video file is required by the API.");
  }

  const formData = new FormData();

  // 1. Append the single video file
  formData.append('video_file', videoFile);

  // 2. Append all context options as individual form fields
  // This matches the `Form()` parameters in your FastAPI endpoint
  const defaultOptions = {
    extract_audio: true,
    sample_fps: 5, // Default to 5 FPS if not provided
    confidence_threshold: 0.3,
    zone_type: 'unknown',
    is_night_time: false,
    enable_alerts: true,
  };

  const finalOptions = { ...defaultOptions, ...options };

  // Append each option to the FormData
  Object.keys(finalOptions).forEach(key => {
    // Convert boolean to string as FastAPI Form() expects
    let value = finalOptions[key];
    if (typeof value === 'boolean') {
      value = value ? 'true' : 'false';
    }
    formData.append(key, value);
  });
  
  // Log the FormData keys for debugging (optional)
  // for (var pair of formData.entries()) {
  //   console.log(pair[0]+ ', ' + pair[1]); 
  // }

  try {
    const response = await fetch(`${API_BASE_URL}/api/process/video`, {
      method: 'POST',
      body: formData,
      // Note: Do NOT set 'Content-Type' manually.
      // The browser automatically sets it for 'multipart/form-data'.
    });

    if (!response.ok) {
      // Try to parse the detailed error message from FastAPI
      let errorMessage = `API Error: ${response.status} ${response.statusText}`;
      try {
        const errorData = await response.json();
        // `detail` is the key FastAPI uses for HTTP error messages
        errorMessage = errorData.detail || JSON.stringify(errorData);
      } catch (e) {
        // Could not parse JSON, use the basic status text
      }
      throw new Error(errorMessage);
    }

    // If successful, return the JSON response from the backend
    return await response.json();

  } catch (error) {
    console.error("Error in analyzeMedia API call:", error);
    // Re-throw the error so the UI component's .catch() block
    // can capture it and set the error state.
    throw error;
  }
};

/**
 * Fetches the health status of the API.
 * This is still valid for server_v3.py.
 */
export const getApiHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    if (!response.ok) {
      throw new Error(`API Health Check Failed: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error fetching API health:", error);
    throw error;
  }
};

/**
 * Fetches recent alerts (for the dashboard).
 * This is still valid for server_v3.py.
 */
export const getRecentAlerts = async () => {
   try {
    const response = await fetch(`${API_BASE_URL}/api/alerts/recent`);
    if (!response.ok) {
      throw new Error(`Failed to fetch recent alerts: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error fetching recent alerts:", error);
    throw error;
  }
};

// =======================================================================
// === NEW FUNCTION ADDED FOR LIVE STREAM PAGE ===
// =======================================================================

/**
 * Analyzes a single image frame by sending it to the backend's frame endpoint.
 *
 * This function is designed to work with server_v3.py's /api/process/frame
 *
 * @param {File} frameFile - The image file (PNG, JPG) to analyze.
 * @returns {Promise<object>} - The FrameAnalysisResponse from the backend.
 */
export const analyzeFrame = async (frameFile) => {
  if (!frameFile) {
    throw new Error("An image file is required for frame analysis.");
  }

  const formData = new FormData();
  formData.append('frame_file', frameFile);

  try {
    const response = await fetch(`${API_BASE_URL}/api/process/frame`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errorMessage = `API Error: ${response.status} ${response.statusText}`;
      try {
        const errorData = await response.json();
        errorMessage = errorData.detail || JSON.stringify(errorData);
      } catch (e) {
        // Could not parse JSON
      }
      throw new Error(errorMessage);
    }

    return await response.json();

  } catch (error) {
    console.error("Error in analyzeFrame API call:", error);
    throw error;
  }
};