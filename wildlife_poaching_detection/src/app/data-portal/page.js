// app/data-portal/page.js
'use client';

import { UploadCloud } from 'lucide-react';

export default function DataPortalPage() {
    
    const handleSubmit = (e) => {
        e.preventDefault();
        // Handle the form submission logic here
        // e.g., new FormData(e.target) and send to an API endpoint
        alert('Data submitted (simulation).');
    };

    return (
        <div className="space-y-8 max-w-4xl mx-auto">
            <div>
                <h2 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">Data Contribution Portal</h2>
                <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
                    Help improve our AI models by submitting your training and testing data.
                </p>
            </div>

            {/* --- Submission Form --- */}
            <div className="bg-white dark:bg-gray-800 p-6 sm:p-8 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
                <form onSubmit={handleSubmit} className="space-y-6">
                    
                    {/* --- File Upload Area --- */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            Upload File
                        </label>
                        <div className="mt-1 flex justify-center px-6 pt-10 pb-12 border-2 border-gray-300 dark:border-gray-600 border-dashed rounded-md">
                            <div className="text-center">
                                <UploadCloud className="mx-auto h-12 w-12 text-gray-400" />
                                <div className="mt-4 flex text-sm text-gray-600 dark:text-gray-400">
                                    <label
                                        htmlFor="file-upload"
                                        className="relative cursor-pointer rounded-md font-medium text-blue-600 dark:text-blue-400 hover:text-blue-500 focus-within:outline-none"
                                    >
                                        <span>Upload a file</span>
                                        <input id="file-upload" name="file-upload" type="file" className="sr-only" />
                                    </label>
                                    <p className="pl-1">or drag and drop</p>
                                </div>
                                <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">Image, Video, or Audio files (up to 50MB)</p>
                            </div>
                        </div>
                    </div>

                    {/* --- Data Classification --- */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <label htmlFor="dataType" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                Data Type
                            </label>
                            <select
                                id="dataType"
                                name="dataType"
                                className="mt-1 block w-full rounded-md border-gray-300 bg-white dark:bg-gray-700 dark:border-gray-600 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                            >
                                <option>Please select...</option>
                                <option value="image">Image</option>
                                <option value="video">Video</option>
                                <option value="audio">Audio</option>
                            </select>
                        </div>
                        
                        <div>
                            <label htmlFor="contentLabel" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                Content Label (Ground Truth)
                            </label>
                            <select
                                id="contentLabel"
                                name="contentLabel"
                                className="mt-1 block w-full rounded-md border-gray-300 bg-white dark:bg-gray-700 dark:border-gray-600 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                            >
                                <option>Please select...</option>
                                <option value="human">Human (No threat)</option>
                                <option value="human_poacher">Human (Poacher/Threat)</option>
                                <option value="animal">Animal (Specify in notes)</option>
                                <option value="sound_gunshot">Sound (Gunshot)</option>
                                <option value="sound_vehicle">Sound (Vehicle)</option>
                                <option value="sound_other">Sound (Other)</option>
                                <option value="other">Other</option>
                            </select>
                        </div>
                    </div>

                    {/* --- Purpose --- */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                            Purpose of Submission
                        </label>
                        <div className="mt-2 space-y-2">
                            <div className="flex items-center">
                                <input id="purpose-train" name="purpose" type="radio" value="training" className="focus:ring-blue-500 h-4 w-4 text-blue-600 border-gray-300" />
                                <label htmlFor="purpose-train" className="ml-3 block text-sm font-medium text-gray-700 dark:text-gray-300">
                                    Training Data (Help teach the model)
                                </label>
                            </div>
                            <div className="flex items-center">
                                <input id="purpose-test" name="purpose" type="radio" value="testing" className="focus:ring-blue-500 h-4 w-4 text-blue-600 border-gray-300" />
                                <label htmlFor="purpose-test" className="ml-3 block text-sm font-medium text-gray-700 dark:text-gray-300">
                                    Testing Data (Validate model accuracy)
                                </label>
                            </div>
                        </div>
                    </div>

                    {/* --- Notes --- */}
                    <div>
                        <label htmlFor="notes" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                            Additional Notes (Optional)
                        </label>
                        <textarea
                            id="notes"
                            name="notes"
                            rows={3}
                            className="mt-1 block w-full rounded-md border-gray-300 bg-white dark:bg-gray-700 dark:border-gray-600 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                            placeholder="e.g., 'Two rhinos, one calf', 'Distant engine sound at 3s', 'Image from a drone'"
                        />
                    </div>

                    {/* --- Submit Button --- */}
                    <div className="text-right">
                        <button
                            type="submit"
                            className="inline-flex justify-center rounded-md border border-transparent bg-blue-600 py-2 px-4 text-sm font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                        >
                            Submit Data
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}