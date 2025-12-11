// Custom SSE handler for smooth progress updates
document.addEventListener('DOMContentLoaded', function () {
    // File Input UX
    const fileInput = document.querySelector('input[type="file"]');
    const fileLabel = document.getElementById('file-label-text');
    const dropZone = document.getElementById('drop-zone');

    if (fileInput) {
        fileInput.addEventListener('change', function (e) {
            if (e.target.files.length > 0) {
                fileLabel.innerHTML = `<span class="text-indigo-400 font-semibold">${e.target.files[0].name}</span>`;
                const icon = dropZone.querySelector('svg');
                if (icon) icon.classList.add('text-indigo-500');
                dropZone.classList.add('border-indigo-500', 'bg-indigo-500/10');
                dropZone.classList.remove('border-gray-700');
            }
        });
    }

    // Listen for HTMX SSE messages
    document.body.addEventListener('htmx:sseBeforeMessage', function (event) {
        try {
            // Parse JSON data from SSE
            const data = JSON.parse(event.detail.data);
            console.log('📥 SSE Data:', data);

            if (data.type === 'progress') {
                // Update progress bar smoothly without replacing DOM
                const progressBar = document.getElementById('progress-bar');
                const progressValue = document.getElementById('progress-value');
                const progressDesc = document.getElementById('progress-desc');
                const queuePosition = document.getElementById('queue-position');
                const queueSize = document.getElementById('queue-size');
                const elapsedTime = document.getElementById('elapsed-time');

                if (progressBar) progressBar.style.width = data.progress + '%';
                if (progressValue) progressValue.textContent = data.progress;
                if (progressDesc) progressDesc.textContent = data.progress_desc;
                if (queuePosition) queuePosition.textContent = data.position;
                if (queueSize) queueSize.textContent = data.queue_size;
                if (elapsedTime) elapsedTime.textContent = data.elapsed_time;

                // Prevent default HTMX swap
                event.preventDefault();
            } else if (data.type === 'success') {
                // Render success template
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

                            <div class="flex justify-center">
                                <a href="/static/${data.output_filename}" download="${data.output_filename}"
                                    class="block w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white font-bold py-4 px-6 rounded-xl transition-all shadow-lg hover:shadow-green-500/25 text-center transform active:scale-95 flex items-center justify-center gap-2">
                                    <span>⬇️ Download Video</span>
                                </a>
                            </div>
                        </div>
                    `;
                }
                event.preventDefault();
            } else if (data.type === 'error') {
                // Render error template
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
                event.preventDefault();
            }
        } catch (e) {
            console.error('Error parsing SSE JSON:', e);
            // Let HTMX handle it as HTML if not valid JSON
        }
    });
});
