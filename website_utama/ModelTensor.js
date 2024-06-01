// Bagian 1 : Load Data dengan papa parse
let trainingData = [];
let testingData = [];

// Fungsi papa.parse untuk memproses file CSV
function loadData(file, isTest = false) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      download: true,
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      transform: function(value, column) {
        if (['EPS', 'PER', 'PBV', 'Harga'].includes(column)) {
          return parseFloat(value.replace(/\./g, '').replace(/,/g, '.'));
        }
        return value;
      },
      complete: results => {
        if (isTest) {
          testingData = results.data;
        } else {
          trainingData = trainingData.concat(results.data);
        }
        resolve();
      },
      error: error => {
        console.error(`Error parsing file ${file}:`, error);
        reject(error);
      }
    });
  });
}

// Bagian 2 : Memisahkan data untuk data training dan data testing 
function checkDataValidity(data) {
  if (data.some(d => isNaN(d.EPS) || isNaN(d.PBV) || isNaN(d.PER))) {
      console.error("Data contains invalid numeric values");
      return false;
  }
  return true;
}

// , dilakukan juga pengecekkan error
async function loadAndPrepareData() {
    const trainingYears = [2017, 2018, 2019, 2020, 2021, 2022];
    const testingYear = 2023;
    try {
        for (let year of trainingYears) {
            await loadData(`/data/${year}.csv`);
        }
        await loadData(`/data/${testingYear}.csv`, true);
  
        // pengecekan data
        if (trainingData.length === 0 || testingData.length === 0) {
            console.error("Tidak ada data training atau data testing");
            return;
        }
    } catch (error) {
        console.error("Gagal Memuat Data:", error);
    }
}
loadAndPrepareData()

// bagian 3 : mengubah data menjadi tensor
function displayData(inputTensor, labelTensor, inputMax, inputMin, labelMax, labelMin) {
  inputTensor.array().then(inputArray => {
      console.log("Input Data:");
      console.log(inputArray);
  });
  labelTensor.array().then(labelArray => {
      console.log("Label Data:");
      console.log(labelArray);
  });

  // normalisasi data untuk mempermudah proses komputasi
  const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
  const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

  normalizedInputs.array().then(normalizedInputArray => {
      console.log("Normalized Input Data:");
      console.log(normalizedInputArray);
  });
  normalizedLabels.array().then(normalizedLabelArray => {
      console.log("Normalized Label Data:");
      console.log(normalizedLabelArray);
  });
}

function convertToTensor(data) {
    return tf.tidy(() => {
        const inputs = data.map(d => [
            parseFloat(d.EPS),
            parseFloat(d.PER),
            parseFloat(d.PBV)
        ]);
        // merancang tensor
        const labels = data.map(d => parseFloat(d.Harga));
        const inputTensor = tf.tensor2d(inputs, [inputs.length, 3]);
        const labelTensor = tf.tensor1d(labels);

        const inputMax = inputTensor.max(0);
        const inputMin = inputTensor.min(0);
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const epsilon = 1e-7;
        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin).add(epsilon));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin).add(epsilon));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        };
    });
}

// Pengecekan Data
async function processAndConvertData() {
  await loadAndPrepareData();
  if (trainingData.length === 0 || testingData.length === 0) {
    console.error("training data atau testing data tidak tersedia");
    return;  
  }
  // Konversi data setelah data dimuat
  const tensorTrainingData = convertToTensor(trainingData);
  const tensorTestingData = convertToTensor(testingData);
  return { tensorTrainingData, tensorTestingData };
}

// bagian 4 : Merancang model
function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [3], units: 250, activation: 'relu', 
  kernelRegularizer: tf.regularizers.l2({l2: 0.01}) 
  }));
  model.add(tf.layers.dense({units: 100, activation: 'relu', 
    kernelRegularizer: tf.regularizers.l2({l2: 0.01}) 
  }));
  model.add(tf.layers.dropout(0.5));
  model.add(tf.layers.dense({units: 64, activation: 'relu'}));
  model.add(tf.layers.dense({units: 1}));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError',
    metrics: ['mse']
  });

  return model;
}

// menyimpan model
async function saveModel(model) {
  try {
    // Menyimpan model ke IndexedDB
    await model.save('indexeddb://my-model');
  } catch (error) {
    console.error('Gagal menyimpan model:', error);
  }
}

// bagian 5 : Melatih model
async function trainModel(model, trainingData, testingData) {
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError',
    metrics: ['mse']
  });

  const { inputs: trainInputs, labels: trainLabels } = trainingData;
  const { inputs: testInputs, labels: testLabels } = testingData;

  return await model.fit(trainInputs, trainLabels, {
    epochs: 2000,
    validationData: [testInputs, testLabels],
    shuffle: true,
    verbose: 1,
    callbacks: [
      tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 10 }),
      new tf.CustomCallback({
        onEpochEnd: async (epoch, logs) => {
          if ((epoch + 1) % 5 === 0) {
            await saveModel(model);
            console.log(`Checkpoint disimpan untuk epoch ${epoch + 1}`);
          }
        }
      })
    ]
  });
}

// eksekusi latihan
async function runTraining() {
  const { tensorTrainingData, tensorTestingData } = await processAndConvertData();
  if (!tensorTrainingData || !tensorTestingData) {
    console.error("Tidak dapat mengkonversi data ke tensor");
    return;
  }

  const model = createModel();
  const history = await trainModel(model, tensorTrainingData, tensorTestingData);
  console.log(history.history);
  await saveModel(model);
}

runTraining();

// bagian 6 : Evaluasi model
async function evaluateModel(model, testingData) {
  const {inputs: testInputs, labels: testLabels} = testingData;

  try {
    const evalResult = await model.evaluate(testInputs, testLabels);
    console.log(`hasil evaluasi - Loss: ${evalResult[0].dataSync()[0]}, MSE: ${evalResult[1].dataSync()[0]}`);

    // Melakukan prediksi dan menangani hasilnya
    const predictions = model.predict(testInputs);
    const predictedLabels = predictions.dataSync();
    const trueLabels = testLabels.dataSync();

    displayEvaluationResults(predictedLabels, trueLabels);
    const rmse = calculateRMSE(predictedLabels, trueLabels);
    console.log(`Root Mean Squared Error (RMSE): ${rmse}`);
  } catch (error) {
    console.error("Error during model evaluation:", error);
  }
}

function calculateRMSE(predictions, labels) {
  const errors = labels.map((label, index) => predictions[index] - label);
  const squaredErrors = errors.map(error => error ** 2);
  const mse = squaredErrors.reduce((sum, squaredError) => sum + squaredError, 0) / errors.length;
  const rmse = Math.sqrt(mse);
  return rmse;
}

function displayEvaluationResults(predictedLabels, trueLabels) {
  const sampleSize = Math.min(predictedLabels.length, 10);
  for (let i = 0; i < sampleSize; i++) {
    console.log(`Sampel ${i + 1}: Prediksi = ${predictedLabels[i]}, Aktual = ${trueLabels[i]}`);
  }
}

// menyiapkan fungsi dan model evaluasi
async function runEvaluation() {
  await loadAndPrepareData();
  if (!testingData || testingData.length === 0) {
    console.error("Data testing tidak tersedia atau tidak berhasil dimuat");
    return;
  }

  try {
    const model = await loadModel();
    if (!model) {
      console.error("Model tidak berhasil dimuat");
      return;
    } 

    const tensorTestingData = convertToTensor(testingData);
    await evaluateModel(model, tensorTestingData);
  } catch (error) {
    console.error("Terjadi kesalahan selama evaluasi model:", error);
  }
}

async function loadModel() {
  try {
    const model = await tf.loadLayersModel('indexeddb://my-model');
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['mse']
    });
    return model;
  } catch (error) {
    console.error('Gagal memuat model:', error);
    return null;
  }
}

// mendapatkan rasio keuangan dari web
async function predictFromLocalStorage() {
  const eps = parseFloat(localStorage.getItem("EPS"));
  const per = parseFloat(localStorage.getItem("PER"));
  const pbv = parseFloat(localStorage.getItem("PBV"));

  if (isNaN(eps) || isNaN(per) || isNaN(pbv)) {
    console.error("Data EPS, PER, atau PBV tidak valid atau tidak tersedia di localStorage");
    return;
  }

  const model = await loadModel();
  if (!model) {
    console.error("Model tidak berhasil dimuat");
    return;
  }
  const predictedPrice = await predictPrice(model, eps, per, pbv);

  document.getElementById('FV').innerHTML = 'Harga Wajar : Rp ' + predictedPrice.toLocaleString('id-ID', {
    style: 'currency',
    currency: 'IDR',
    minimumFractionDigits: 0
  }).replace(/^IDR/, '');
}

// memanggil fungsi setelah memuat halaman
document.addEventListener('DOMContentLoaded', predictFromLocalStorage);

// Fungsi input Manual
async function predictPrice(model, eps, per, pbv) {
  const inputTensor = tf.tensor2d([[eps, per, pbv]], [1, 3]);

  const inputMax = tf.tensor([1000000, 50, 50]); 
  const inputMin = tf.tensor([0, 0, 0]); 
  const epsilon = 1e-7;

  // Normalisasi input
  const normalizedInput = inputTensor.sub(inputMin).div(inputMax.sub(inputMin).add(epsilon));

  // Lakukan prediksi
  const prediction = model.predict(normalizedInput);
  prediction.print();

  // Jika output juga dinormalisasi, lakukan denormalisasi
  const labelMax = tf.tensor(10000);  
  const labelMin = tf.tensor(0);  
  const denormalizedOutput = prediction.mul(labelMax.sub(labelMin)).add(labelMin);
  
  const predictedPrice = denormalizedOutput.dataSync()[0];

  // Menyimpan hasil prediksi ke web
  localStorage.setItem("HargaWajar", predictedPrice);

  updateDecision();
  return predictedPrice;  // Pastikan untuk mengembalikan nilai yang diprediksi
}
