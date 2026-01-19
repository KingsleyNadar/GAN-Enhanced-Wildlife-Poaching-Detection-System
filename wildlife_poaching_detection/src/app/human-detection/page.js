'use client';
import { useState } from 'react';
import { analyzeMedia } from '../../lib/api'; // This now points to our updated function

// --- UI Components ---
const ActionButton = ({ onClick, isLoading, children }) => (
  <button
    onClick={onClick}
    disabled={isLoading}
    className="w-full mt-4 px-4 py-3 text-sm font-semibold text-white bg-indigo-600 rounded-lg shadow-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all duration-200"
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

// --- [NEW] GAN PREDICTION COMPONENT ---
// This new component formats and displays the gan_prediction object

const getGanAlertStyle = (riskAssessment = "") => {
    const assessment = riskAssessment.toUpperCase();
    if (assessment.includes("CRITICAL")) {
        return { bg: 'bg-red-100 dark:bg-red-900/40', text: 'text-red-800 dark:text-red-200', border: 'border-red-500' };
    }
    if (assessment.includes("HIGH")) {
        return { bg: 'bg-yellow-100 dark:bg-yellow-800/40', text: 'text-yellow-800 dark:text-yellow-200', border: 'border-yellow-500' };
    }
    if (assessment.includes("MEDIUM")) {
        // Using blue for MEDIUM to distinguish from HIGH (yellow)
        return { bg: 'bg-blue-100 dark:bg-blue-800/40', text: 'text-blue-800 dark:text-blue-200', border: 'border-blue-500' };
    }
    return { bg: 'bg-gray-100 dark:bg-gray-700/40', text: 'text-gray-800 dark:text-gray-300', border: 'border-gray-500' };
};

const GanPredictionCard = ({ prediction }) => {
    // Get the color-coding based on the risk assessment string
    const styles = getGanAlertStyle(prediction.risk_assessment);

    return (
        <div className={`p-4 rounded-lg border-l-4 ${styles.bg} ${styles.border}`}>
            <h5 className={`font-bold ${styles.text}`}>{prediction.risk_assessment}</h5>
            <p className={`text-sm mt-2 ${styles.text}`}>
                <span className="font-semibold">Recommended Action:</span> {prediction.recommended_action}
            </p>
            <div className={`mt-3 pt-2 border-t ${styles.border} border-opacity-30 grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1 text-xs ${styles.text} opacity-90`}>
                <p><span className="font-semibold">Operation Type:</span> {prediction.operation_type} ({prediction.pattern_description})</p>
                <p><span className="font-semibold">Escalation:</span> {prediction.escalation_pattern}</p>
                <p><span className="font-semibold">Confidence:</span> {(prediction.confidence * 100).toFixed(1)}%</p>
                <p><span className="font-semibold">Model:</span> {prediction.model_version}</p>
            </div>
        </div>
    );
};
// --- [END NEW COMPONENT] ---


export default function HumanDetectionPage() {
  // File and result states
  const [videoFile, setVideoFile] = useState(null);
  const [preview, setPreview] = useState('https://placehold.co/600x400/e2e8f0/94a3b8?text=Your+Video+Here');
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Model options states
  const [zoneType, setZoneType] = useState('unknown');
  const [isNightTime, setIsNightTime] = useState(false);
  const [sample_fps, setSample_fps] = useState(5);

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

    const options = { 
      zone_type: zoneType, 
      is_night_time: isNightTime, 
      sample_fps: sample_fps,
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
        <h2 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">Human & Threat Detection</h2>
        <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
          Upload a video to run the full analysis pipeline. Audio is extracted automatically.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* --- Controls Column --- */}
        <div className="lg:col-span-1 space-y-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold border-b pb-3 mb-4 dark:text-white dark:border-gray-600">1. Upload Video</h3>
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Video File*</label>
                    <input type="file" accept="video/*" onChange={handleVideoFileChange} className="mt-1 text-sm w-full file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 dark:file:bg-gray-700 dark:file:text-gray-200 dark:hover:file:bg-gray-600"/>
                </div>
            </div>

            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold border-b pb-3 mb-4 dark:text-white dark:border-gray-600">2. Set Context</h3>
                <div>
                    <label htmlFor="zoneType" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Zone Type</label>
                    <select id="zoneType" value={zoneType} onChange={e => setZoneType(e.target.value)} className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white">
                        <option>unknown</option>
                        <option>restricted</option>
                        <option>protected</option>
                        <option>buffer</option>
                        <option>tourist</option>
                    </select>
                </div>
                 <div className="mt-4 space-y-2">
                    <label className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300">
                      <input type="checkbox" checked={isNightTime} onChange={e => setIsNightTime(e.target.checked)} className="h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500" />
                      <span className="ml-2">Is Night Time?</span>
                    </label>
                </div>
                <div className="mt-4">
                     <label htmlFor="sample_fps" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Sample FPS ({sample_fps})</label>
                     <input id="sample_fps" type="range" min="1" max="30" value={sample_fps} onChange={e => setSample_fps(parseInt(e.target.value))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700" />
                </div>
            </div>
            
             <ActionButton onClick={handleRunModel} isLoading={isLoading}>Analyze Media</ActionButton>
        </div>

        {/* --- Results Column (Updated) --- */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold border-b pb-3 mb-4 dark:text-white dark:border-gray-600">3. Analysis Results</h3>
          {isLoading && <div className="text-center py-10">Processing, please wait...</div>}
          {error && <div className="p-4 bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300 rounded-lg text-sm">{error}</div>}
          {!isLoading && !results && <div className="text-center py-10 text-gray-500">Upload a video and click Analyze to see results.</div>}
          
          {results && (
            <div className="space-y-6">
                {/* 1. Event Alerts */}
                <div>
                    <h4 className="font-semibold text-gray-800 dark:text-gray-200">Event Alerts</h4>
                    {results.threat_alerts && results.threat_alerts.length > 0 ? (
                        <div className="mt-2 space-y-3">
                            {results.threat_alerts.map((alert, i) => <AlertCard key={i} alert={alert} />)}
                        </div>
                    ) : <p className="mt-2 text-sm text-gray-500">No specific event alerts triggered.</p>}
                </div>

                 {/* 2. [NEW] GAN Threat Assessment */}
                 {results.gan_prediction && (
                    <div>
                        <h4 className="font-semibold text-gray-800 dark:text-gray-200">AI Threat Assessment (GAN)</h4>
                        <div className="mt-2">
                            {/* This is the new component being used */}
                            <GanPredictionCard prediction={results.gan_prediction} />
                        </div>
                    </div>
                )}
                
                 {/* 3. Visual Detections */}
                 <div>
                    <h4 className="font-semibold text-gray-800 dark:text-gray-200">Visual Detections (Phase 2)</h4>
                    <div className="relative mt-2">
                        <video src={preview} controls className="w-full h-auto object-contain rounded-lg bg-gray-100 dark:bg-gray-900" />
                    </div>
                    {results.phase2_detections && results.phase2_detections.flatMap(f => f.detections).length > 0 ? (
                        <div className="flex flex-wrap gap-2 mt-2">
                        {[...new Set(results.phase2_detections.flatMap(f => f.detections.map(d => d.label)))].map(label => (
                            <span key={label} className="px-2 py-1 text-xs font-medium text-indigo-800 bg-indigo-100 rounded-full dark:bg-indigo-900 dark:text-indigo-300">{label}</span>
                        ))}
                        </div>
                    ) : <p className="mt-2 text-sm text-gray-500">No human/threat objects detected.</p>}
                </div>

                {/* 4. Audio Analysis */}
                {results.audio_segments && results.audio_segments.length > 0 && (
                    <div>
                         <h4 className="font-semibold text-gray-800 dark:text-gray-200">Audio Analysis</h4>
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
                )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}