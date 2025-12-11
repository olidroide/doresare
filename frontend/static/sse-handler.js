/**
 * SSE Handler for Real-time Progress Updates
 * Automatically initializes when the SSE progress container is found
 */
(function () {
    'use strict';

    /**
     * Initialize SSE connection for a job
     * @param {string} jobId - The job ID to monitor
     * @param {HTMLElement} container - The container element with progress UI
     */
    function initializeSSE(jobId, container) {
        const eventSource = new EventSource(`/status/${jobId}`);

        console.log('🔌 SSE Connection established for job:', jobId);

        eventSource.onmessage = function (event) {
            console.log('📥 SSE Raw Data:', event.data);

            try {
                const data = JSON.parse(event.data);
                console.log('📥 SSE Parsed Data:', data);
                console.log('🎯 Event Type:', data.type);

                if (data.type === 'progress') {
                    handleProgressUpdate(data, container);
                } else if (data.type === 'success') {
                    handleSuccess(data);
                    eventSource.close();
                } else if (data.type === 'error') {
                    handleError(data);
                    eventSource.close();
                }
            } catch (e) {
                console.error('❌ Error parsing SSE JSON:', e);
            }
        };

        eventSource.onerror = function (error) {
            console.error('❌ SSE Connection error:', error);
            eventSource.close();
        };
    }

    /**
     * Handle progress updates
     */
    function handleProgressUpdate(data, container) {
        console.log('🔄 Updating progress UI...');

        // Query elements within the container to avoid conflicts with #loading
        const progressBar = container.querySelector('#progress-bar');
        const progressValue = container.querySelector('#progress-value');
        const progressDesc = container.querySelector('#progress-desc');
        const queuePosition = container.querySelector('#queue-position');
        const queueSize = container.querySelector('#queue-size');
        const elapsedTime = container.querySelector('#elapsed-time');

        console.log('📍 DOM Elements found:', {
            progressBar: !!progressBar,
            progressValue: !!progressValue,
            progressDesc: !!progressDesc,
            queuePosition: !!queuePosition,
            queueSize: !!queueSize,
            elapsedTime: !!elapsedTime
        });

        if (progressBar) {
            const width = data.progress + '%';
            progressBar.style.width = width;
            console.log('✅ Updated progress bar to:', width);
        }
        if (progressValue) progressValue.textContent = data.progress;
        if (progressDesc) {
            progressDesc.textContent = data.progress_desc || 'Processing...';
            console.log('✅ Updated progress desc to:', data.progress_desc);
        }

        // Queue position display
        if (queuePosition && queueSize) {
            if (data.position === 'Processing' || data.queue_size === '-') {
                // Just show "Processing" without the slash
                queuePosition.textContent = 'Processing';
                queueSize.textContent = '';
            } else {
                // Show position/total
                queuePosition.textContent = data.position || '-';
                queueSize.textContent = data.queue_size || '-';
            }
        }

        if (elapsedTime) {
            const formattedTime = data.elapsed_time ? data.elapsed_time + 's' : '0s';
            elapsedTime.textContent = formattedTime;
            console.log('✅ Updated elapsed time to:', formattedTime);
        }
    }

    /**
     * Handle success event
     */
    function handleSuccess(data) {
        const resultado = document.getElementById('resultado');
        if (resultado) {
            resultado.innerHTML = `
                <div class="glass-panel p-6 rounded-2xl border-green-500/30 animate-fade-in shadow-lg shadow-green-900/20">
                    <h3 class="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-emerald-500 mb-6 text-center">✨ Video Completed!</h3>
                    
                    <div class="relative rounded-xl overflow-hidden shadow-2xl border border-gray-800 bg-black aspect-video mb-6">
                        <video controls class="w-full h-full object-contain">
                            <source src="/static/${data.output_filename}" type="video/mp4">
                            Your browser does not support HTML5 video.
                        </video>
                    </div>

                    <div class="flex flex-col gap-3">
                        <a href="/static/${data.output_filename}" download="${data.output_filename}"
                            class="block w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white font-bold py-4 px-6 rounded-xl transition-all shadow-lg hover:shadow-green-500/25 text-center transform active:scale-95 flex items-center justify-center gap-2">
                            <span>⬇️ Download Video</span>
                        </a>
                        
                        <button onclick="resetToUploadState()" 
                            class="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold py-4 px-6 rounded-xl transition-all shadow-lg hover:shadow-indigo-500/25 transform active:scale-95 flex items-center justify-center gap-2">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd" />
                            </svg>
                            <span>Upload New Video</span>
                        </button>
                    </div>
                </div>
            `;
        }
    }

    /**
     * Handle error event
     */
    function handleError(data) {
        const resultado = document.getElementById('resultado');
        if (resultado) {
            resultado.innerHTML = `
                <div class="glass-panel p-6 rounded-2xl border-red-500/30 border">
                    <div class="flex items-center gap-3 mb-3 text-red-400">
                        <h3 class="text-lg font-bold">❌ Error</h3>
                    </div>
                    <p class="text-gray-300 leading-relaxed">${data.error_message}</p>
                </div>
            `;
        }
    }

    /**
     * Auto-initialize when container is found
     * Uses MutationObserver to detect when queued.html is loaded into the DOM
     */
    function autoInit() {
        const container = document.getElementById('sse-progress-container');
        if (container) {
            const jobId = container.dataset.jobId;
            if (jobId) {
                console.log('🚀 SSE Handler: Found container with job ID:', jobId);
                initializeSSE(jobId, container);
            } else {
                console.error('❌ SSE Handler: Container found but no job ID in data-job-id attribute');
            }
        }
    }

    /**
     * Setup MutationObserver to watch for dynamic content
     */
    function setupObserver() {
        // Only setup if body exists
        if (!document.body) {
            console.warn('⚠️ SSE Handler: document.body not ready yet');
            return;
        }

        const observer = new MutationObserver(function (mutations) {
            mutations.forEach(function (mutation) {
                mutation.addedNodes.forEach(function (node) {
                    if (node.nodeType === 1) { // Element node
                        // Check if the added node is the container or contains it
                        if (node.id === 'sse-progress-container' || node.querySelector('#sse-progress-container')) {
                            autoInit();
                        }
                    }
                });
            });
        });

        // Start observing the document body for changes
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    // Initialize everything when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () {
            autoInit();
            setupObserver();
        });
    } else {
        autoInit();
        setupObserver();
    }
})();
