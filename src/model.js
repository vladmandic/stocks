// @ts-nocheck
/* global tf, tfvis */

// import * as tf from '@tensorflow/tfjs'; // <https://js.tensorflow.org/api/latest/>

const int = 255;
const sub = 0; // 0.5;
const mul = 0.75; // 1.5;

let model;
let stats;

async function dispose() {
  if (model) {
    model.layers.forEach((layer) => layer.dispose);
    model.weights.forEach((weight) => weight.dispose);
    if (model.optimizer) model.optimizer.dispose();
    model.dispose();
  }
}

async function createCNN(params) { // concept based on <https://www.hindawi.com/journals/complexity/2020/6622927/>
  // eslint-disable-next-line no-console
  console.log('Model create CNN:', params);

  // console.log('Create start', tf.engine().memory().numTensors);
  await tf.engine().startScope();
  model = tf.sequential({ name: 'sequentialStocks' });

  // TBD

  await tf.engine().endScope();
  // console.log('Create end', tf.engine().memory().numTensors);
}

async function createRNN(params) { // concept based on <https://www.codeproject.com/Articles/1265477/TensorFlow-js-Predicting-Time-Series-Using-Recurre>
  // eslint-disable-next-line no-console
  console.log('Model create RNN:', params);

  // console.log('Create start', tf.engine().memory().numTensors);
  await tf.engine().startScope();
  model = tf.sequential({ name: 'sequentialStocks' });

  // model definition
  model.add(tf.layers.dense({ // https://js.tensorflow.org/api/latest/#layers.dense
    name: 'layerDenseInput',
    batchInputShape: [1, params.inputWindow],
    trainable: true,
    units: params.neurons,

    kernelInitializer: params.kernelInitializer || 'glorotNormal',
    kernelConstraint: params.constraint,
    biasInitializer: params.biasInitializer || 'glorotNormal',
    biasConstraint: params.constraint,
    unitForgetBias: params.forgetBias || false,
  }));

  model.add(tf.layers.reshape({ // https://js.tensorflow.org/api/latest/#layers.reshape
    name: 'layerReshape',
    trainable: false,
    targetShape: [params.features, (params.neurons / params.features)],
  }));

  const cells = [];
  for (let index = 0; index < params.layers; index++) {
    cells.push(tf.layers[params.cells]({ // https://js.tensorflow.org/api/latest/#Layers-Recurrent
      name: `cell${params.cells}${index}`,
      trainable: true,
      dtype: params.dtype || 'float32',
      units: params.neurons,

      activation: params.activation || 'tanh',
      kernelInitializer: params.kernelInitializer || 'glorotNormal',
      kernelConstraint: params.constraint,
      recurrentInitializer: params.kernelInitializer || 'glorotNormal',
      recurrentActivation: params.recurrentActivation || 'hardSigmoid',
      recurrentConstraint: params.constraint,
      biasInitializer: params.biasInitializer || 'glorotNormal',
      biasConstraint: params.constraint,
      unitForgetBias: params.forgetBias || false,
    }));
  }

  model.add(tf.layers.rnn({ // https://js.tensorflow.org/api/latest/#layers.rnn
    name: 'layerRNN',
    trainable: true,
    cell: cells,
    returnSequences: false,
    returnState: false,
    statefull: !params.shuffle || true,
    dtype: params.dtype,
  }));

  model.add(tf.layers.dense({ // https://js.tensorflow.org/api/latest/#layers.dense
    name: 'layerDenseOutput',
    units: 1, // params.outputWindow,
    dtype: params.dtype || 'float32',
    trainable: true,

    kernelInitializer: params.kernelInitializer || 'glorotNormal',
    kernelConstraint: params.constraint,
    biasInitializer: params.biasInitializer || 'glorotNormal',
    biasConstraint: params.constraint,
    unitForgetBias: params.forgetBias || false,
  }));

  // compile model
  // const rate = params.dtype === 'int32' ? params.learningRate * int : params.learningRate;
  model.compile({ // https://js.tensorflow.org/api/latest/#Training-Optimizers
    optimizer: tf.train[params.optimizer](params.learningRate),
    loss: params.loss || 'meanSquaredError',
    metrics: ['accuracy'],
  });
  await tf.engine().endScope();
  // console.log('Create end', tf.engine().memory().numTensors);
}

async function train(input, output, params, callback) {
  // if (!model) await create(params);
  await dispose();
  await createRNN(params);

  // console.log('Normalize start', tf.engine().memory().numTensors);
  await tf.engine().startScope();
  // normalize inputs
  const max = Math.max(...output);
  const min = Math.min(...output);
  const inputT = params.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(input, [input.length, params.inputWindow]).sub(min).mul(int).div(max - min).toInt())
    : tf.tidy(() => tf.tensor2d(input, [input.length, params.inputWindow]).sub(min).div(max - min).sub(sub).mul(mul));
  const outputT = params.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(output, [output.length, params.outputWindow]).sub(min).mul(int).div(max - min).toInt())
    : tf.tidy(() => tf.tensor2d(output, [output.length, params.outputWindow]).sub(min).div(max - min).sub(sub).mul(mul));
  // console.log('Normalize end', tf.engine().memory().numTensors);

  if (tfvis && params.visor) tfvis.show.valuesDistribution({ name: 'Input Data Distribution', tab: 'Visor' }, inputT);
  if (tfvis && params.visor) tfvis.show.valuesDistribution({ name: 'Output Data Distribution', tab: 'Visor' }, outputT);

  const batchLogs = [];
  function visorPlot(batch, logs) {
    if (!params.visor) return;
    batchLogs.push(logs);
    const values = batchLogs.map((log) => ({ x: log.batch, y: log.loss }));
    tfvis.render.linechart({ name: 'Batch Loss', tab: 'Visor' }, { values, series: ['loss'] });
  }

  // used by fit callback
  function fitEpochCallback(epoch, logs) {
    const loss = Math.trunc(1000 * Math.sqrt(logs.loss) / (params.dtype === 'int32' ? int : 1)) / 1000;
    if (loss < params.targetLoss) {
      model.stopTraining = true;
      callback(epoch, loss, `Fit early stop: epoch: ${epoch} loss: ${loss}`);
    } else {
      callback(epoch, loss);
    }
  }
  // execute fit with callback
  // console.log('Fit start', tf.engine().memory().numTensors);
  stats = await model.fit(inputT, outputT, // https://js.tensorflow.org/api/latest/#tf.LayersModel.fit
    { batchSize: params.inputWindow,
      epochs: params.epochs,
      validationSplit: params.validationSplit,
      shuffle: params.shuffle || false,
      callbacks: {
        onBatchEnd: (batch, logs) => visorPlot(batch, logs),
        onEpochEnd: (epoch, logs) => fitEpochCallback(epoch, logs),
      },
    });
  // console.log('Fit end', tf.engine().memory().numTensors);
  stats.params = params;
  stats.max = Math.trunc(10000 * max) / 10000;
  stats.min = Math.trunc(10000 * min) / 10000;
  stats.length = output.length;

  // console.log('Eval start', tf.engine().memory().numTensors);
  const evaluateT = model.evaluate(inputT, outputT, { batchSize: params.inputWindow });
  inputT.dispose();
  outputT.dispose();
  if (evaluateT) {
    stats.eval = Math.trunc(100000 * evaluateT[0].dataSync()[0]) / 1000;
    stats.accuracy = Math.trunc(100000 * evaluateT[1].dataSync()[0]) / 1000;
    evaluateT[0].dispose();
    evaluateT[1].dispose();
  }
  await tf.engine().endScope();
  // console.log('Eval end', tf.engine().memory().numTensors);
}

async function predict(input) {
  // console.log('Predict start', tf.engine().memory().numTensors);
  await tf.engine().startScope();
  if (!stats || !stats.params) return null;
  const inputT = stats.params.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(input, [1, input.length]).sub(stats.min).mul(int).div(stats.max - stats.min).toInt())
    : tf.tidy(() => tf.tensor2d(input, [1, input.length]).sub(stats.min).div(stats.max - stats.min).sub(sub).mul(mul));
  const outputT = model.model.predict(inputT, { batchSize: stats.params.inputWindow });
  const normalizeT = stats.params.dtype === 'int32'
    ? tf.tidy(() => outputT.mul(stats.max - stats.min).div(int).add(stats.min))
    : tf.tidy(() => outputT.div(mul).add(sub).mul(stats.max - stats.min).add(stats.min));
  const output = normalizeT.dataSync();
  inputT.dispose();
  normalizeT.dispose();
  outputT.dispose();
  await tf.engine().endScope();
  // console.log('Predict end', tf.engine().memory().numTensors);
  return output;
}

export { train, predict, model, stats };
