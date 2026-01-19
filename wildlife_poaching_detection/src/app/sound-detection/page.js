'use client';
import { useState } from 'react';
// We can now use our new, unified api library!
import { analyzeMedia } from '../../lib/api'; 

// --- UI Components (Re-usable) ---
const ActionButton = ({ onClick, isLoading, children }) => (
  <button
    onClick={onClick}
    disabled={isLoading}
    className="w-full mt-4 px-4 py-3 text-sm font-semibold text-white bg-green-600 rounded-lg shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all duration-200"
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
// --- End of UI Components ---


// Renaming component to be more accurate
export default function SoundDetectionPage() {
  // --- STATE ---
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // --- HANDLERS ---
  const handleVideoFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setVideoFile(selectedFile);
      setResults(null);
      setError(null);
      setVideoPreview(URL.createObjectURL(selectedFile));
    }
  };
  
  // UPDATED: Using the new analyzeMedia function
  const handleRunModel = async () => {
    if (!videoFile) return;
    setIsLoading(true);
    setError(null);
    setResults(null);

    // We only care about audio, but must still pass video.
    // We can use default options.
    const options = {
        extract_audio: true 
    };

    try {
      // Use the new, unified API library
      const data = await analyzeMedia(videoFile, options);
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // --- RENDER ---
  return (
    <div className="space-y-8">
      <div>
        {/* Updated titles */}
        <h2 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">Sound Detection</h2>
        <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
          Upload a video file to analyze its audio track for threats.
        </p>
      </div>

       <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* --- Controls Column --- */}
        <div className="lg:col-span-1 space-y-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold border-b pb-3 mb-4 dark:text-white dark:border-gray-600">1. Upload Video</h3>
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Video File*</label>
                    <input 
                      type="file" 
                      accept="video/*" 
                      onChange={handleVideoFileChange} 
                      className="mt-1 text-sm w-full file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-green-50 file:text-green-700 hover:file:bg-green-100 dark:file:bg-gray-700 dark:file:text-gray-200 dark:hover:file:bg-gray-600"
                    />
                </div>
            </div>
             <ActionButton onClick={handleRunModel} isLoading={isLoading}>Analyze Audio</ActionButton>
        </div>

        {/* --- Results Column (Focusing on Audio) --- */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold border-b pb-3 mb-4 dark:text-white dark:border-gray-600">2. Analysis Results</h3>
           <div className="min-h-[300px] flex flex-col justify-center">
            {isLoading && <div className="text-center py-10">Processing, please wait...</div>}
            {error && <div className="p-4 bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300 rounded-lg text-sm">{error}</div>}
            
            {!isLoading && !results && (
                <div className="text-center text-gray-500">
                    {videoPreview ? (
                        <video src={videoPreview} controls className="w-full mt-4 rounded-lg" />
                    ) : (
                        <div className="w-full h-24 bg-gray-100 dark:bg-gray-700 rounded-lg flex items-center justify-center">
                            Your video preview will appear here
                        </div>
                    )}
                    <p className="mt-2 text-sm">Upload a video file to analyze its audio.</p>
                </div>
            )}
            
            {/* UPDATED: Results section focused on audio */}
            {results && (
              <div className="space-y-6">
                  {/* UPDATED: results.threat_alerts */}
                  <div>
                      <h4 className="font-semibold text-gray-800 dark:text-gray-200">Alerts</h4>
                      {results.threat_alerts && results.threat_alerts.length > 0 ? (
                          <div className="mt-2 space-y-3">
                              {results.threat_alerts.map((alert, i) => <AlertCard key={i} alert={alert} />)}
                          </div>
                      ) : <p className="mt-2 text-sm text-gray-500">No critical threats detected.</p>}
                  </div>

                  {/* UPDATED: results.audio_segments (now a list) */}
                   {results.audio_segments && results.audio_segments.length > 0 ? (
                    <div>
                         <h4 className="font-semibold text-gray-800 dark:text-gray-200">Audio Classification</h4>
                         <div className="mt-2 space-y-2">
                            {results.audio_segments.map((seg, i) => (
                                <div key={i} className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                                    <p className="text-sm text-gray-600 dark:text-gray-300">
                                        <span className="font-bold">{seg.label}</span> detected at {seg.start_time.toFixed(1)}s - {seg.end_time.toFixed(1)}s
                                    </p>
                                    <p className="text-xs text-gray-500">Confidence: {(seg.confidence * 100).toFixed(1)}%</p>
                                </div>
                            ))}
                         </div>
                    </div>
                  ) : (
                    <p className="mt-2 text-sm text-gray-500">No audio segments were processed or detected.</p>
                  )}
              </div>
            )}
           </div>
        </div>
      </div>
    </div>
  );
}