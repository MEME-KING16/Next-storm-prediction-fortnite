const createModel = () => {
    const model = tf.sequential();

    // Input layer
    model.add(tf.layers.dense({ inputShape: [3], units: 32, activation: 'relu' }));
    
    // Hidden layers
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    
    // Output layer
    model.add(tf.layers.dense({ units: 3 }));

    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError'
    });

    return model;
};
const model = createModel();


// Generate some training data
const x = []
const y = []
x.push([206.5, 288.5, 200.3000030517578], [165.5, 300.5, 160.8000030517578], [130.5, 317.5, 114.9000015258789])
y.push([165.5, 300.5, 160.8000030517578], [130.5, 317.5, 114.9000015258789], [107.5, 321.5, 91.5])
x.push([235.5, 294.5, 200.39999389648438], [224.5, 325.5, 158.39999389648438], [189.5, 338.5, 103.69999694824219], [184.5, 329.5, 79.5], [193.5, 339.5, 49.79999923706055])
y.push([224.5, 325.5, 158.39999389648438], [189.5, 338.5, 103.69999694824219], [184.5, 329.5, 79.5], [193.5, 339.5, 49.79999923706055], [147.5, 354.5, 29.700000762939453])
x.push([244.5, 305.5, 199.60000610351562],[211.5, 327.5, 157.39999389648438], [177.5, 319.5, 116.4000015258789], [176.5, 305.5, 90.80000305175781], [198.5, 294.5, 52.5])
y.push([211.5, 327.5, 157.39999389648438],[177.5, 319.5, 116.4000015258789], [176.5, 305.5, 90.80000305175781], [198.5, 294.5, 52.5], [252.5, 299.5, 30.899999618530273])
const xs = tf.tensor2d(x);
const ys = tf.tensor2d(y);

let loass = []
// Train the model
async function trainModel() {
    await model.fit(xs, ys, {
        epochs: 2000, // Start with 500 epochs
        validationSplit: 0.2, // Helps with evaluation
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                loass.push(logs.loss)
                console.log(`Epoch: ${epoch}, Loss: ${logs.loss}`);
            }
        }
    });
    document.getElementById("upload").disabled = false
    console.log('Training complete');
}

trainModel();

