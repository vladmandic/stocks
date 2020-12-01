import * as tf from '@tensorflow/tfjs';

const int = 255;
const sub = 0.5;
const mul = 1.5;

async function train(input, output, params, callback) {
  // console.log('Training:', params, input, output);
  const model = tf.sequential();

  // normalize inputs
  const max = Math.max(...output);
  const min = Math.min(...output);
  const inputT = params.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(input, [input.length, params.inputWindow]).sub(min).mul(int).div(max - min).toInt())
    : tf.tidy(() => tf.tensor2d(input, [input.length, params.inputWindow]).sub(min).div(max - min).sub(sub).mul(mul));
  const outputT = params.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(output, [output.length, params.outputWindow]).sub(min).mul(int).div(max - min).toInt())
    : tf.tidy(() => tf.tensor2d(output, [output.length, params.outputWindow]).sub(min).div(max - min).sub(sub).mul(mul));

  // check normalization
  // const t = outputT.dataSync();
  // console.log(Math.min(...t), Math.max(...t));

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
    cells.push(tf.layers[params.cells]({ // https://js.tensorflow.org/api/latest/#Layers-Recurrent
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
        // onBatchEnd: (batch, logs) => console.log(batch, logs),
        onEpochEnd: (epoch, logs) => normalizeLoss(epoch, logs),
      },
    });
  stats.params = params;
  stats.max = max;
  stats.min = min;

  const evaluateT = model.evaluate(inputT, outputT, { batchSize: params.inputWindow });
  stats.eval = Math.trunc(100000 * evaluateT.dataSync()[0]) / 1000;
  inputT.dispose();
  outputT.dispose();
  evaluateT.dispose();
  return { model, stats };
}

async function predict(model, input) {
  const inputT = model.stats.params.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(input, [1, input.length]).sub(model.stats.min).mul(int).div(model.stats.max - model.stats.min).toInt())
    : tf.tidy(() => tf.tensor2d(input, [1, input.length]).sub(model.stats.min).div(model.stats.max - model.stats.min).sub(sub).mul(mul));
  const outputT = model.model.predict(inputT, { batchSize: model.stats.params.inputWindow });
  const normalizeT = model.stats.params.dtype === 'int32'
    ? tf.tidy(() => outputT.mul(model.stats.max - model.stats.min).div(int).add(model.stats.min))
    : tf.tidy(() => outputT.div(mul).add(sub).mul(model.stats.max - model.stats.min).add(model.stats.min));
  const output = normalizeT.dataSync();
  inputT.dispose();
  normalizeT.dispose();
  outputT.dispose();
  return output;
}

export { train, predict };
