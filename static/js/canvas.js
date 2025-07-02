const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const clearButton = document.getElementById('clearButton');
const mainPredictionDiv = document.getElementById('mainPrediction');
const top3PredictionsList = document.getElementById('top3Predictions');

let isDrawing = false;
let timeout = null;

// Set up canvas for drawing
ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.strokeStyle = 'white'; // Draw in white on a black background
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);

function sendCanvasForPrediction() {
    if (timeout) {
        clearTimeout(timeout);
    }
    timeout = setTimeout(() => {
        const imageData = canvas.toDataURL('image/png');
        fetch('/predict_live', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `image_data=${encodeURIComponent(imageData)}`,
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                mainPredictionDiv.textContent = `Error: ${data.error}`;
                top3PredictionsList.innerHTML = '';
            } else {
                mainPredictionDiv.textContent = data.prediction;
                top3PredictionsList.innerHTML = '';
                data.top3.forEach(item => {
                    const li = document.createElement('li');
                    li.textContent = `Digit ${item[0]} (Confidence: ${(item[1] * 100).toFixed(2)}%)`;
                    top3PredictionsList.appendChild(li);
                });
            }
        })
        .catch(error => {
            console.error('Error:', error);
            mainPredictionDiv.textContent = 'Prediction Error';
            top3PredictionsList.innerHTML = '';
        });
    }, 200); // Debounce to avoid too many requests
}

canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});

canvas.addEventListener('mouseup', () => {
    isDrawing = false;
    sendCanvasForPrediction();
});

canvas.addEventListener('mouseout', () => {
    isDrawing = false;
});

clearButton.addEventListener('click', () => {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    mainPredictionDiv.textContent = '';
    top3PredictionsList.innerHTML = '';
});