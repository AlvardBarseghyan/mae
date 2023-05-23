async function handleClick(event) {
    const overlayImages = document.querySelectorAll('.overlay-image');
    const overlay1 = overlayImages[0];
    const rect = overlay1.getBoundingClientRect();

    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const response = await fetch('/update_images', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ x, y })
    });

    const jsonResponse = await response.json();
    if (jsonResponse.success) {
        const timestamp = new Date().getTime();
        overlayImages.forEach((overlayImage) => {
            const currentSrc = overlayImage.getAttribute('src');
            const updatedSrc = currentSrc.split('?')[0] + '?t=' + timestamp;
            overlayImage.setAttribute('src', updatedSrc);
        });
    }
}

// app.js
function addCacheBreaker() {
    const baseImages = document.querySelectorAll('.base-image');
    const timestamp = new Date().getTime();
    
    baseImages.forEach((baseImage) => {
        const currentSrc = baseImage.getAttribute('src');
        const updatedSrc = currentSrc + '?t=' + timestamp;
        baseImage.setAttribute('src', updatedSrc);
    });
}

addCacheBreaker();
