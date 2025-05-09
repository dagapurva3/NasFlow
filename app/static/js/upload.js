document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData();
    const fileInput = document.getElementById('dataset');
    const progressBar = document.querySelector('.progress-bar');
    const statusMessage = document.getElementById('status-message');
    
    // Check if file is selected
    if (fileInput.files.length === 0) {
        statusMessage.textContent = 'Please select a file';
        statusMessage.style.color = 'red';
        document.getElementById('progress').style.display = 'block';
        return;
    }
    
    const file = fileInput.files[0];
    formData.append('dataset', file);
    
    // Display file info
    document.getElementById('progress').style.display = 'block';
    statusMessage.textContent = `Uploading ${file.name} (${formatFileSize(file.size)})...`;
    statusMessage.style.color = 'black';
    progressBar.style.width = '0%';
    progressBar.textContent = '0%';
    
    try {
        // Create XMLHttpRequest to track upload progress
        const xhr = new XMLHttpRequest();
        
        // Track upload progress
        xhr.upload.addEventListener('progress', (event) => {
            if (event.lengthComputable) {
                const percentComplete = (event.loaded / event.total) * 100;
                progressBar.style.width = percentComplete + '%';
                progressBar.textContent = Math.round(percentComplete) + '%';
                
                if (percentComplete < 100) {
                    statusMessage.textContent = `Uploading: ${Math.round(percentComplete)}% (${formatFileSize(event.loaded)} of ${formatFileSize(event.total)})`;
                } else {
                    statusMessage.textContent = 'Processing file, please wait...';
                }
            }
        });
        
        // Handle response
        xhr.addEventListener('load', function() {
            if (xhr.status >= 200 && xhr.status < 300) {
                const data = JSON.parse(xhr.responseText);
                window.location.href = data.redirect;
            } else {
                const errorMsg = xhr.responseText ? JSON.parse(xhr.responseText).error : 'Unknown error';
                throw new Error(errorMsg);
            }
        });
        
        // Handle errors
        xhr.addEventListener('error', function() {
            throw new Error('Network error occurred');
        });
        
        // Open and send request
        xhr.open('POST', '/upload', true);
        xhr.send(formData);
        
    } catch (error) {
        progressBar.style.width = '0%';
        statusMessage.textContent = `Error: ${error.message}`;
        statusMessage.style.color = 'red';
    }
});

// Format file size to human-readable format
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}