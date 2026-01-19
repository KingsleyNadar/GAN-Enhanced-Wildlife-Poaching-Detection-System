'use client';

import { useState } from 'react';
import { Upload, Loader2, AlertCircle } from 'lucide-react';
// 1. Import your new API function
import { analyzeFrame } from '../../lib/api'; // Adjust path if /lib is not at root

const YOUTUBE_VIDEO_ID = 'F0GOOP82094'; // The ID from the URL

export default function LiveStreamPage() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileChange = (e) => {
        setSelectedFile(e.target.files[0]);
        setAnalysisResult(null);
        setError(null);
    };

    // 2. This function is now much simpler!
    const handleAnalyzeFrame = async () => {
        if (!selectedFile) {
            setError('Please select a frame to analyze first.');
            return;
        }

        setIsLoading(true);
        setError(null);
        setAnalysisResult(null);

        try {
            // It just calls your API library function
            const data = await analyzeFrame(selectedFile);
            setAnalysisResult(data);

        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    // ... (rest of the LiveStreamPage component is unchanged) ...
    // (renderDetections function, return() statement, etc.)
    
    // Helper to format detections for display
    const renderDetections = (detections) => {
        if (!detections || detections.length === 0) {
            return <p className="text-gray-500 dark:text-gray-400">No detections.</p>;
        }
        return (
            <ul className="list-disc pl-5 space-y-1">
                {detections.map((det, index) => (
                    <li key={index}>
                        <span className="font-semibold text-gray-900 dark:text-white">{det.label}:</span>
                        <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">
                            {(det.confidence * 100).toFixed(1)}% confidence
                        </span>
                    </li>
                ))}
            </ul>
        );
    };

    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">Live Stream Analysis</h2>
                <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
                    Monitor the 24/7 wildlife feed and manually upload frames for analysis.
                </p>
            </div>

            {/* --- Live Stream Embed --- */}
            <div className="bg-white dark:bg-gray-800 p-4 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
                <div className="aspect-w-16 aspect-h-9">
                    <iframe
                        src={`https://www.youtube.com/embed/${YOUTUBE_VIDEO_ID}?autoplay=1&mute=1`}
                        title="YouTube video player"
                        frameBorder="0"
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowFullScreen
                        className="w-full h-full rounded-lg"
                        style={{ minHeight: '480px' }}
                    ></iframe>
                </div>
            </div>

            {/* --- Frame Analysis Section --- */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* --- Upload Card --- */}
                <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Analyze a Frame</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                        Pause the stream, take a screenshot, and upload it here to run the AI model.
                    </p>
                    
                    <input
                        type="file"
                        id="frame-upload"
                        accept="image/png, image/jpeg"
                        onChange={handleFileChange}
                        className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                    />
                    
                    <button
                        onClick={handleAnalyzeFrame}
                        disabled={isLoading || !selectedFile}
                        className="mt-6 w-full inline-flex items-center justify-center rounded-md border border-transparent bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isLoading ? (
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        ) : (
                            <Upload className="mr-2 h-4 w-4" />
                        )}
                        {isLoading ? 'Analyzing...' : 'Analyze Frame'}
                    </button>
                    
                    {error && (
                        <div className="mt-4 flex items-center text-sm text-red-600 dark:text-red-400">
                            <AlertCircle className="h-4 w-4 mr-2" />
                            {error}
                        </div>
                    )}
                </div>

                {/* --- Results Card --- */}
                <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Analysis Results</h3>
                    
                    {isLoading && (
                        <div className="flex items-center justify-center h-24">
                            <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
                        </div>
                    )}

                    {analysisResult && (
                        <div className="space-y-4">
                            <div>
                                <h4 className="font-semibold text-blue-700 dark:text-blue-300">Phase 1 (Animals)</h4>
                                {renderDetections(analysisResult.phase1_detections)}
                            </div>
                            <div>
                                <h4 className="font-semibold text-green-700 dark:text-green-300">Phase 2 (Human/Threats)</h4>
                                {renderDetections(analysisResult.phase2_detections)}
                            </div>
                            <p className="text-xs text-gray-500 dark:text-gray-400 pt-2 border-t dark:border-gray-700">
                                Processing time: {analysisResult.processing_time.toFixed(2)} seconds.
                            </p>
                        </div>
                    )}

                    {!isLoading && !analysisResult && !error && (
                        <p className="text-gray-500 dark:text-gray-400">Upload a frame to see the analysis results here.</p>
                    )}
                </div>
            </div>
        </div>
    );
}