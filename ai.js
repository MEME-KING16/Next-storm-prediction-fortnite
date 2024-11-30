// Create a simple model in TensorFlow.js
const createModel = () => {
    const model = tf.sequential();

    // Input layer with 3 features
    model.add(tf.layers.dense({ inputShape: [3], units: 512, activation: 'relu' }));
    model.add(tf.layers.batchNormalization());

    // Add hidden layers with regularization
    model.add(tf.layers.dense({
        units: 1024,
        activation: 'swish',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
    }));
    model.add(tf.layers.dropout({ rate: 0.2 }));

    model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    model.add(tf.layers.batchNormalization());

    // Output layer with 3 outputs: x, y, and r
    model.add(tf.layers.dense({ units: 3 }));

    // Compile the model with a custom loss function
    const customLoss = (yTrue, yPred) => {
        const weights = tf.tensor([1.0, 1.0, 0.5]); // Emphasize x and y
        return tf.losses.meanSquaredError(yTrue.mul(weights), yPred.mul(weights));
    };

    model.compile({
        optimizer: 'adam',
        loss: customLoss,
        metrics: ['mae']
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

// Train the model
async function trainModel() {
    await model.fit(xs, ys, {
        epochs: 7000
    });
    model.save("downloads://")
    console.log('Training complete');
}

trainModel();

