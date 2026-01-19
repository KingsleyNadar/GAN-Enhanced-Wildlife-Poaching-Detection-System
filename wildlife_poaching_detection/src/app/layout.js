'use client';

import { useState, useEffect, useRef } from 'react';
import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { Inter } from 'next/font/google';
// All icons, including the new ones
import { 
    Bot, Home, User, PawPrint, Ear, Menu, 
    Pin, PinOff, Sun, Moon, FileUp, Youtube 
} from 'lucide-react';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

// --- Helper to get page title from URL ---
const getTitle = (pathname) => {
    switch (pathname) {
        case '/':
            return 'Poaching Detection';
        case '/human-detection':
            return 'Human Detection';
        case '/animal-detection':
            return 'Animal Detection';
        case '/sound-detection':
            return 'Sound Detection';
        // New page titles
        case '/data-portal':
            return 'Data Contribution Portal';
        case '/live-stream':
            return 'Live Stream Analysis';
        default:
            return 'Poaching Detection';
    }
};


// --- Main Components ---
const Sidebar = ({ isSidebarOpen }) => {
    const pathname = usePathname();
    const navItems = [
        { href: '/', icon: <Home className="h-5 w-5" />, label: 'Dashboard' },
        { href: '/human-detection', icon: <User className="h-5 w-5" />, label: 'Human Detection' },
        { href: '/animal-detection', icon: <PawPrint className="h-5 w-5" />, label: 'Animal Detection' },
        { href: '/sound-detection', icon: <Ear className="h-5 w-5" />, label: 'Sound Detection' },
        // New nav items
        { href: '/data-portal', icon: <FileUp className="h-5 w-5" />, label: 'Data Portal' },
        { href: '/live-stream', icon: <Youtube className="h-5 w-5" />, label: 'Live Stream' },
    ];

    return (
        <aside 
            className={`absolute top-0 left-0 h-full w-64 bg-gray-900 text-white transform ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} transition-transform duration-300 ease-in-out`}
        >
            <div className="flex items-center gap-2 px-4 h-16 border-b border-gray-800">
                <Bot className="h-6 w-6" />
                <h1 className="text-xl font-bold">Wilderness Manager</h1>
            </div>
            <nav className="p-4 space-y-2">
                {navItems.map(item => (
                    <Link
                        key={item.href}
                        href={item.href}
                        className={`flex items-center gap-3 rounded-lg px-3 py-2 transition-all ${pathname === item.href ? 'bg-gray-700 text-white' : 'text-gray-400 hover:bg-gray-800 hover:text-white'}`}
                    >
                        {item.icon}
                        {item.label}
                    </Link>
                ))}
            </nav>
        </aside>
    );
};

const Header = ({ title, onMenuClick, onThemeToggle, isDarkMode, isPinned }) => (
    <header className="flex h-16 items-center justify-between gap-4 border-b bg-white px-6 dark:bg-gray-900/50 dark:border-gray-800">
        <div className="flex items-center gap-4">
            <button
                className="p-2 -ml-2 rounded-md text-gray-600 dark:text-gray-300"
                onClick={onMenuClick}
                aria-label="Toggle sidebar pin"
            >
                {isPinned ? <PinOff className="h-6 w-6" /> : <Pin className="h-6 w-6" />}
            </button>
             <h1 className="text-xl font-semibold text-gray-900 dark:text-white">{title}</h1>
        </div>
        
        <div className="flex items-center gap-4">
            <button onClick={onThemeToggle} className="p-2 rounded-full text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800">
                {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </button>
        </div>
    </header>
);

export default function RootLayout({ children }) {
    const [isSidebarOpen, setSidebarOpen] = useState(false);
    const [isPinned, setIsPinned] = useState(true); // Changed to true
    const [isDarkMode, setIsDarkMode] = useState(false);
    const pathname = usePathname();
    const title = getTitle(pathname);
    const hoverTimeoutRef = useRef(null);

    useEffect(() => {
        const savedTheme = localStorage.getItem('theme');
        setIsDarkMode(savedTheme === 'dark');
    }, []);

    useEffect(() => {
        if (isDarkMode) {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }, [isDarkMode]);
    
    useEffect(() => {
        if (isSidebarOpen && window.innerWidth < 768) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = 'auto';
        }
    }, [isSidebarOpen]);

    const handleMouseEnter = () => {
        clearTimeout(hoverTimeoutRef.current);
        if (!isPinned) {
            setSidebarOpen(true);
        }
    };

    const handleMouseLeave = () => {
        if (!isPinned) {
            hoverTimeoutRef.current = setTimeout(() => {
                setSidebarOpen(false);
            }, 300);
        }
    };

    const handlePinToggle = () => {
        const newPinState = !isPinned;
        setIsPinned(newPinState);
        if (newPinState) {
            setSidebarOpen(true);
        }
    };

    const toggleTheme = () => {
        localStorage.setItem('theme', isDarkMode ? 'light' : 'dark');
        setIsDarkMode(!isDarkMode);
    };
    
    const shouldSidebarBeOpen = isPinned || isSidebarOpen;

    return (
        <html lang="en">
            <body className={`${inter.className} bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-50`}>
                <div className="flex min-h-screen">
                    {/* --- UNIFIED HOVER AREA & SIDEBAR CONTAINER --- */}
                    <div 
                        onMouseEnter={handleMouseEnter}
                        onMouseLeave={handleMouseLeave}
                        className={`fixed top-0 left-0 h-full z-40 transition-all duration-300 ease-in-out ${shouldSidebarBeOpen ? 'w-64' : 'w-0 md:w-4'}`}
                    >
                        <Sidebar isSidebarOpen={shouldSidebarBeOpen} />
                    </div>
                    
                    {/* Mobile overlay */}
                    {shouldSidebarBeOpen && !isPinned && (
                         <div className="fixed inset-0 bg-black/60 z-30 md:hidden" onClick={() => setSidebarOpen(false)}></div>
                    )}

                    {/* Main content */}
                    <div className={`flex flex-1 flex-col transition-all duration-300 ease-in-out ${shouldSidebarBeOpen ? 'md:ml-64' : 'ml-0'}`}>
                        <Header 
                            title={title} 
                            onMenuClick={handlePinToggle} 
                            onThemeToggle={toggleTheme} 
                            isDarkMode={isDarkMode}
                            isPinned={isPinned}
                        />
                        <main className="flex-1 p-6 sm:p-8">
                            {children}
                        </main>
                    </div>
                </div>
            </body>
        </html>
    );
}