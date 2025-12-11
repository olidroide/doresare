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

    // Hide upload form when processing starts
    // HTMX triggers this event before showing the indicator
    document.body.addEventListener('htmx:beforeRequest', function (event) {
        const uploadFormContainer = document.getElementById('upload-form-container');
        if (uploadFormContainer) {
            // Fade out animation
            uploadFormContainer.style.transition = 'opacity 0.3s ease-out, transform 0.3s ease-out';
            uploadFormContainer.style.opacity = '0';
            uploadFormContainer.style.transform = 'scale(0.95)';

            setTimeout(() => {
                uploadFormContainer.style.display = 'none';
            }, 300);
        }
    });
});

// Global function to reset to initial state (show upload form)
function resetToUploadState() {
    const uploadFormContainer = document.getElementById('upload-form-container');
    const resultado = document.getElementById('resultado');

    // Clear resultado
    if (resultado) {
        resultado.innerHTML = '';
    }

    // Show upload form with fade in
    if (uploadFormContainer) {
        uploadFormContainer.style.display = 'block';
        // Force reflow
        uploadFormContainer.offsetHeight;
        uploadFormContainer.style.opacity = '1';
        uploadFormContainer.style.transform = 'scale(1)';

        // Reset file input
        const fileInput = uploadFormContainer.querySelector('input[type="file"]');
        const fileLabel = document.getElementById('file-label-text');
        const dropZone = document.getElementById('drop-zone');

        if (fileInput) {
            fileInput.value = '';
        }
        if (fileLabel) {
            fileLabel.innerHTML = 'Tap to upload video';
        }
        if (dropZone) {
            const icon = dropZone.querySelector('svg');
            if (icon) icon.classList.remove('text-indigo-500');
            dropZone.classList.remove('border-indigo-500', 'bg-indigo-500/10');
            dropZone.classList.add('border-gray-700');
        }
    }
}

