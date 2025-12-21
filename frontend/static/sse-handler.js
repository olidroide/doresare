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
        const videoContainer = document.getElementById('video-result-container');
        const videoPlayer = document.getElementById('result-video');
        const downloadBtn = document.getElementById('download-btn');

        console.log('✅ Job completed. Updating result UI...');

        if (videoPlayer) {
            const videoUrl = data.video_url || (data.output_filename ? `/static/${data.output_filename}` : '');
            if (videoUrl) {
                videoPlayer.src = videoUrl;
                if (videoContainer) videoContainer.classList.remove('hidden');
                videoPlayer.load();
                if (downloadBtn) {
                    downloadBtn.href = videoUrl;
                }
            }
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
