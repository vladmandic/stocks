import * as tf from '@tensorflow/tfjs';

const int = 255;
const sub = 0;

async function train(input, output, params, callback) {
  // console.log('Training:', params, input, output);
  const model = tf.sequential();

  // normalize inputs
  const max = Math.max(...output);
  const inputT = params.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(input, [input.length, params.inputWindow]).mul(int).div(max).toInt())
    : tf.tidy(() => tf.tensor2d(input, [input.length, params.inputWindow]).div(max).sub(sub));
  const outputT = params.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(output, [output.length, params.outputWindow]).mul(int).div(max).toInt())
    : tf.tidy(() => tf.tensor2d(output, [output.length, params.outputWindow]).div(max).sub(sub));

  // model definition
  model.add(tf.layers.dense({ // https://js.tensorflow.org/api/latest/#layers.dense
    name: 'layerDenseInput',
    units: params.neurons,
    unitForgetBias: params.forgetBias || false,
    kernelInitializer: params.kernelInitializer || 'glorotNormal',
    biasInitializer: params.biasInitializer || 'glorotNormal',
    batchInputShape: [1, params.inputWindow],
  }));

  model.add(tf.layers.reshape({ // https://js.tensorflow.org/api/latest/#layers.reshape
    name: 'layerReshape',
    targetShape: [params.features, Math.trunc(params.neurons / params.features)],
  }));

  const cells = [];
  for (let index = 0; index < params.layers; index++) {
    cells.push(tf.layers[params.cells]({ // https://js.tensorflow.org/api/latest/#layers.lstmCell
      name: `cell${params.cells}${index}`,
      recurrentActivation: params.recurrentActivation || 'hardSigmoid',
      units: params.neurons,
      activation: params.activation || 'tanh',
      kernelInitializer: params.kernelInitializer || 'glorotNormal',
      recurrentInitializer: params.kernelInitializer || 'glorotNormal',
      dtype: params.dtype || 'float32',
      unitForgetBias: params.forgetBias || false,
    }));
  }

  model.add(tf.layers.rnn({ // https://js.tensorflow.org/api/latest/#layers.rnn
    name: 'layerRNN',
    cell: cells,
    returnSequences: false,
    dtype: params.dtype,
  }));

  model.add(tf.layers.dense({ // https://js.tensorflow.org/api/latest/#layers.dense
    name: 'layerDenseOutput',
    units: 1, // params.outputWindow,
    kernelInitializer: params.kernelInitializer || 'glorotNormal',
    dtype: params.dtype || 'float32',
  }));

  // compile model
  // const rate = params.dtype === 'int32' ? params.learningRate * int : params.learningRate;
  model.compile({ // https://js.tensorflow.org/api/latest/#Training-Optimizers
    optimizer: tf.train[params.optimizer](params.learningRate),
    loss: params.loss || 'meanSquaredError',
    // metrics: ['accuracy'],
  });

  // used by fit callback
  function normalizeLoss(epoch, logs) {
    // console.log('onEpochEnd', epoch, logs);
    const loss = Math.trunc(1000 * Math.sqrt(logs.loss) / (params.dtype === 'int32' ? int : 1)) / 1000;
    callback(epoch, loss);
  }

  // execute fit with callback
  const stats = await model.fit(inputT, outputT, // https://js.tensorflow.org/api/latest/#tf.LayersModel.fit
    { batchSize: params.inputWindow,
      epochs: params.epochs,
      validationSplit: params.validationSplit,
      shuffle: params.shuffle || false,
      callbacks: {
        // onTrainBegin: (logs) => console.log('onTrainBegin', logs),
        onEpochEnd: (epoch, logs) => normalizeLoss(epoch, logs),
        // onTrainEnd: (logs) => console.log('onTrainEnd', logs),
        // onBatchEnd: (batch, logs) => console.log(batch, logs),
      },
    });
  stats.params = params;
  stats.max = max;

  const evaluateT = model.evaluate(inputT, outputT, { batchSize: params.inputWindow });
  stats.eval = Math.trunc(100000 * evaluateT.dataSync()[0]) / 1000;
  inputT.dispose();
  outputT.dispose();
  evaluateT.dispose();
  return { model, stats };
}

async function predict(model, input) {
  const inputT = model.stats.params.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(input, [1, input.length]).mul(int).div(model.stats.max).toInt())
    : tf.tidy(() => tf.tensor2d(input, [1, input.length]).div(model.stats.max).sub(sub));
  const outputT = model.model.predict(inputT, { batchSize: model.stats.params.inputWindow });
  const normalizeT = model.stats.params.dtype === 'int32'
    ? tf.tidy(() => outputT.mul(model.stats.max).div(int))
    : tf.tidy(() => outputT.add(sub).mul(model.stats.max));
  const output = normalizeT.dataSync();
  inputT.dispose();
  normalizeT.dispose();
  outputT.dispose();
  return output;
}

export { train, predict };
