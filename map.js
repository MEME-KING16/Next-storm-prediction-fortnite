// Initialize the Leaflet map
const map = L.map("map", {
    crs: L.CRS.Simple,
    minZoom: -1,
    maxZoom: 3,
    scrollWheelZoom: false,
    smoothWheelZoom: true,
    smoothSensitivity: 1.5,
    zoomControl: false,
    attributionControl: false
});

// Set default bounds for the map
let bounds = [
    [0, 0],
    [500, 500]
];

// Load the default Fortnite map image
let imageOverlay = L.imageOverlay("https://media.fortniteapi.io/images/map.png?showPOI=true", bounds).addTo(map);
map.fitBounds(bounds);

// Function to resize and crop the image to 500x500
function resizeAndCropImage(image, callback) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 500;
    canvas.height = 500;

    const aspectRatio = image.width / image.height;
    let sx = 0, sy = 0, sw = image.width, sh = image.height;

    // Crop the image based on aspect ratio
    if (aspectRatio > 1) { // Image is wider than it is tall
        sw = image.height;
        sx = (image.width - image.height) / 2;
    } else if (aspectRatio < 1) { // Image is taller than it is wide
        sh = image.width;
        sy = (image.height - image.width) / 2;
    }

    // Draw the cropped and resized image onto the canvas
    ctx.drawImage(image, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);

    const resizedImage = new Image();
    resizedImage.src = canvas.toDataURL();
    resizedImage.onload = function() {
        callback(resizedImage);
    };
}

// Function to detect the zone circle in the resized image
function detectZone(image) {
    const canvas = document.getElementById('canvasOutput');
    const ctx = canvas.getContext('2d');
    canvas.width = image.width;
    canvas.height = image.height;
    ctx.drawImage(image, 0, 0);

    const src = cv.imread(canvas);

    // Increase contrast
    const alpha = 2.0; // Contrast control
    const beta = -80;  // Brightness control
    const contrast = new cv.Mat();
    src.convertTo(contrast, -1, alpha, beta);

    const gray = new cv.Mat();
    cv.cvtColor(contrast, gray, cv.COLOR_RGBA2GRAY);

    // Apply binary thresholding to isolate white areas
    const thresholded = new cv.Mat();
    cv.threshold(gray, thresholded, 200, 255, cv.THRESH_BINARY);

    // Apply stronger Gaussian blur to further reduce noise
    const blurred = new cv.Mat();
    cv.GaussianBlur(thresholded, blurred, new cv.Size(7, 7), 2);

    // Apply Canny edge detection to detect the circle's outline with adjusted thresholds
    const edges = new cv.Mat();
    cv.Canny(blurred, edges, 30, 100);  // Lower thresholds to detect more edges

    // Optional: Display the edges for debugging purposes
    cv.imshow('canvasOutput', edges);

    // Detect circles using HoughCircles with refined parameters
    const circles = new cv.Mat();
    cv.HoughCircles(edges, circles, cv.HOUGH_GRADIENT, 1, 700, 100, 50, 0, 0);

    // Check if circles were detected
    if (circles.rows > 0) {
        const circleData = circles.data32F;
        const x = circleData[0];
        const y = circleData[1];
        const r = circleData[2];

        console.log(`Detected zone center: (${x}, ${y}), Radius: ${r}`);

        // Scale the coordinates to fit the map
        const scaleX = 500 / image.width;
        const scaleY = 500 / image.height;
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        const scaledR = r * scaleX;
        const invertedY = 500 - scaledY; // Adjust if Leaflet expects bottom-left origin

        // Draw the detected circle on the map
        L.circle([invertedY, scaledX], {
            radius: scaledR,
            color: 'white',
            weight: 2,
            fillOpacity: 0.1
        }).addTo(map);

        // Ensure the TensorFlow model is loaded before making predictions
        if (typeof model !== 'undefined') {
            let zonepredx = x
            let zonepredy = y
            let zonepredr = r
            const input = tf.tensor2d([[x, y, r]]);
            const output = model.predict(input);

            // Draw the predicted circle on the map
            const outputData = output.dataSync();
            L.circle([500 - outputData[1], outputData[0]], {
                radius: outputData[2],
                color: 'purple',
                weight: 2,
                fillOpacity: 0.1
            }).addTo(map);
            let zonenum = Number(document.getElementById("zonenum").value)
            for (let index = 0; index < 11-zonenum; index++) {
                const inp = tf.tensor2d([[zonepredx,zonepredy,zonepredr]]);
            const out = model.predict(inp);

            // Draw the predicted circle on the map
            const outData = out.dataSync();
            L.circle([500 - outData[1], outData[0]], {
                radius: outData[2],
                color: 'purple',
                weight: 2,
                fillOpacity: 0.1
            }).addTo(map);
             zonepredx = outData[0]
             zonepredy = outData[1]
             zonepredr = outData[2]
             console.log(outData[0],outData[1],outData[2])            
            }
        } else {
            console.error("TensorFlow model is not loaded.");
        }
    } else {
        console.log("No circles detected.");
    }

    // Clean up to save memory
    src.delete();
    gray.delete();
    blurred.delete();
    edges.delete();
    circles.delete();
    thresholded.delete();
    contrast.delete();
}




// Handle image upload and zone detection
document.getElementById('upload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        const img = new Image();
        img.src = e.target.result;
        img.onload = function() {
            resizeAndCropImage(img, detectZone);
        };
    };

    reader.readAsDataURL(file);
});
