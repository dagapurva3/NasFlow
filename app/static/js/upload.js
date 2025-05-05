document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData();
    const fileInput = document.getElementById('dataset');
    const progressBar = document.querySelector('.progress-bar');
    const statusMessage = document.getElementById('status-message');
    
    formData.append('dataset', fileInput.files[0]);
    
    document.getElementById('progress').style.display = 'block';
    statusMessage.textContent = 'Uploading...';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(await response.text());
        }

        const data = await response.json();
        window.location.href = data.redirect;
        
    } catch (error) {
        progressBar.style.width = '0%';
        statusMessage.textContent = `Error: ${error.message}`;
        statusMessage.style.color = 'red';
    }
});