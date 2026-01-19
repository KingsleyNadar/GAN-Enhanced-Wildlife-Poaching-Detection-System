'use client';
import Link from 'next/link';
import { BarChart as BarChartIcon, Users, Annoyed } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// --- Dummy Data for the Chart ---
const weeklyAlertData = [
  { name: 'Mon', Human: 4, Animal: 24, Sound: 2 },
  { name: 'Tue', Human: 3, Animal: 18, Sound: 5 },
  { name: 'Wed', Human: 8, Animal: 32, Sound: 1 },
  { name: 'Thu', Human: 5, Animal: 25, Sound: 8 },
  { name: 'Fri', Human: 7, Animal: 41, Sound: 3 },
  { name: 'Sat', Human: 11, Animal: 55, Sound: 10 },
  { name: 'Sun', Human: 2, Animal: 15, Sound: 4 },
];


// --- UI Components ---
const StatCard = ({ title, value, icon, description }) => (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-start">
            <div className="space-y-1">
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</p>
                <p className="text-3xl font-bold text-gray-900 dark:text-white">{value}</p>
            </div>
            <div className="p-3 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                {icon}
            </div>
        </div>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">{description}</p>
    </div>
);

const AlertChart = () => (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Weekly Alert Summary</h3>
        <div className="h-80 w-full">
            {/* Note: Ensure you have 'recharts' installed for this component to work */}
            <ResponsiveContainer width="100%" height="100%">
                <BarChart data={weeklyAlertData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(128, 128, 128, 0.2)" />
                    <XAxis dataKey="name" tick={{ fill: '#9ca3af' }} fontSize={12} />
                    <YAxis tick={{ fill: '#9ca3af' }} fontSize={12} />
                    <Tooltip
                        cursor={{ fill: 'rgba(128, 128, 128, 0.1)' }} // Changed highlight color
                        contentStyle={{
                            backgroundColor: 'hsl(var(--background))',
                            borderColor: 'hsl(var(--border))',
                            borderRadius: '0.5rem',
                            boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)'
                        }}
                    />
                    <Legend iconSize={10} wrapperStyle={{ fontSize: '14px', color: '#9ca3af' }}/>
                    <Bar dataKey="Human" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Animal" fill="#16a34a" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Sound" fill="#facc15" radius={[4, 4, 0, 0]} />
                </BarChart>
            </ResponsiveContainer>
        </div>
    </div>
);


export default function DashboardPage() {
    return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">Dashboard</h2>
                <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
                    Welcome back! Here's a summary of the detection activity.
                </p>
            </div>

            {/* --- Stats Grid --- */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                <StatCard title="Total Alerts Today" value="12" icon={<BarChartIcon className="h-6 w-6 text-blue-600 dark:text-blue-400" />} description="+20% from yesterday" />
                <StatCard title="Human Detections" value="4" icon={<Users className="h-6 w-6 text-blue-600 dark:text-blue-400" />} description="In restricted zones" />
                <StatCard title="Animal Detections" value="32" icon={<Annoyed className="h-6 w-6 text-blue-600 dark:text-blue-400" />} description="Mostly non-critical" />
            </div>

            {/* --- Chart --- */}
            <AlertChart />

            {/* --- Quick Actions --- */}
            <div>
                 <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Quick Actions</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Link href="/human-detection" className="block p-6 bg-blue-50 dark:bg-blue-900/50 hover:bg-blue-100 dark:hover:bg-blue-900 rounded-xl transition">
                        <h4 className="font-semibold text-blue-800 dark:text-blue-200">Analyze for Humans</h4>
                        <p className="text-sm text-blue-600 dark:text-blue-400 mt-1">Upload video to detect human presence and weapons.</p>
                    </Link>
                    <Link href="/animal-detection" className="block p-6 bg-green-50 dark:bg-green-900/50 hover:bg-green-100 dark:hover:bg-green-900 rounded-xl transition">
                         <h4 className="font-semibold text-green-800 dark:text-green-200">Analyze for Animals</h4>
                        <p className="text-sm text-green-600 dark:text-green-400 mt-1">Upload video to monitor animal activity.</p>
                    </Link>
                     <Link href="/sound-detection" className="block p-6 bg-yellow-50 dark:bg-yellow-900/50 hover:bg-yellow-100 dark:hover:bg-yellow-900 rounded-xl transition">
                         <h4 className="font-semibold text-yellow-800 dark:text-yellow-200">Analyze Sound</h4>
                        <p className="text-sm text-yellow-600 dark:text-yellow-400 mt-1">Upload audio to detect suspicious sounds.</p>
                    </Link>
                </div>
            </div>
        </div>
    );
}

