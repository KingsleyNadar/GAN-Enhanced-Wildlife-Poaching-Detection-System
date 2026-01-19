'use client';
import { useState } from 'react';
import { analyzeMedia } from '../../lib/api'; // This now points to our updated function

// --- UI Components (Re-usable) ---
const ActionButton = ({ onClick, isLoading, children }) => (
  <button
    onClick={onClick}
    disabled={isLoading}
    className="w-full mt-4 px-4 py-3 text-sm font-semibold text-white bg-blue-600 rounded-lg shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all duration-200"
  >
    {isLoading ? 'Analyzing...' : children}
  </button>
);

// Updated AlertCard to match server_v3.py response (alert.type instead of alert.rule)
const AlertCard = ({ alert }) => {
    const alertStyles = {
        CRITICAL: { bg: 'bg-red-100 dark:bg-red-900/40', text: 'text-red-800 dark:text-red-200', border: 'border-red-500' },
        HIGH: { bg: 'bg-yellow-100 dark:bg-yellow-800/40', text: 'text-yellow-800 dark:text-yellow-200', border: 'border-yellow-500' },
        MEDIUM: { bg: 'bg-blue-100 dark:bg-blue-800/40', text: 'text-blue-800 dark:text-blue-200', border: 'border-blue-500' },
        LOW: { bg: 'bg-gray-100 dark:bg-gray-700/40', text: 'text-gray-800 dark:text-gray-300', border: 'border-gray-500' },
    };
    const styles = alertStyles[alert.level] || alertStyles.LOW;

    return (
        <div className={`p-4 rounded-lg border-l-4 ${styles.bg} ${styles.border}`}>
            <div className={`font-bold ${styles.text}`}>{alert.level}: {alert.type.replace(/_/g, ' ')}</div>
            <p className={`text-sm mt-1 ${styles.text}`}>Confidence: {(alert.confidence * 100).toFixed(1)}%</p>
            {alert.timestamp && <p className={`text-xs mt-1 ${styles.text}`}>Timestamp: {alert.timestamp.toFixed(2)}s</p>}
        </div>
    );
};

export default function AnimalDetectionPage() {
  // File and result states
  const [videoFile, setVideoFile] = useState(null);
  const [preview, setPreview] = useState('https://placehold.co/600x400/e2e8f0/94a3b8?text=Your+Video+Here');
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // REMOVED: habitatNearby state

  const handleVideoFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setVideoFile(selectedFile);
      setResults(null);
      setError(null);
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleRunModel = async () => {
    if (!videoFile) return;
    setIsLoading(true);
    setError(null);
    setResults(null);
    
    // We can pass simplified options. 
    // The api.js function will use defaults for missing values.
    const options = {
      zone_type: 'protected' // Example: hardcode a default for this page
    };

    try {
      const data = await analyzeMedia(videoFile, options);
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">Animal Detection</h2>
        <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
          Upload a video to detect animals.
        </p>
      </div>

       <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* --- Controls Column --- */}
        <div className="lg:col-span-1 space-y-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold border-b pb-3 mb-4 dark:text-white dark:border-gray-600">1. Upload Video</h3>
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Video File*</label>
                    <input type="file" accept="video/*" onChange={handleVideoFileChange} className="mt-1 text-sm w-full file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 dark:file:bg-gray-700 dark:file:text-gray-200 dark:hover:file:bg-gray-600"/>
                </div>
                 {/* REMOVED: Habitat Nearby checkbox div */}
            </div>
             <ActionButton onClick={handleRunModel} isLoading={isLoading}>Analyze for Animals</ActionButton>
        </div>

        {/* --- Results Column (Updated) --- */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold border-b pb-3 mb-4 dark:text-white dark:border-gray-600">2. Analysis Results</h3>
          <div className="min-h-[300px] flex flex-col justify-center">
            {isLoading && <div className="text-center py-10">Processing, please wait...</div>}
            {error && <div className="p-4 bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300 rounded-lg text-sm">{error}</div>}
            {!isLoading && !results && (
              <div className="text-center text-gray-500">
                <video src={preview} controls muted loop className="w-full h-auto object-contain rounded-lg bg-gray-100 dark:bg-gray-900" />
                <p className="mt-2 text-sm">Upload a video and click Analyze.</p>
              </div>
            )}
            
            {results && (
              <div className="space-y-6">
                  {/* UPDATED: results.threat_alerts */}
                  <div>
                      <h4 className="font-semibold text-gray-800 dark:text-gray-200">Alerts</h4>
                      {results.threat_alerts && results.threat_alerts.length > 0 ? (
                          <div className="mt-2 space-y-3">
                              {results.threat_alerts.map((alert, i) => <AlertCard key={i} alert={alert} />)}
                          </div>
                      ) : <p className="mt-2 text-sm text-gray-500">No specific animal-related alerts triggered.</p>}
                  </div>
                   {/* UPDATED: results.phase1_detections */}
                   <div>
                       <h4 className="font-semibold text-gray-800 dark:text-gray-200">Detected Animals (Phase 1 Model)</h4>
                        {results.phase1_detections && results.phase1_detections.flatMap(f => f.detections).length > 0 ? (
                          <div className="flex flex-wrap gap-2 mt-2">
                            {[...new Set(results.phase1_detections.flatMap(f => f.detections.map(d => d.label)))].map(label => (
                                <span key={label} className="px-2 py-1 text-xs font-medium text-blue-800 bg-blue-100 rounded-full dark:bg-blue-900 dark:text-blue-300">{label}</span>
                            ))}
                          </div>
                        ) : <p className="mt-2 text-sm text-gray-500">No animals detected.</p>}
                   </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}