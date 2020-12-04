/* global tf, tfvis */

// import * as tf from '@tensorflow/tfjs'; // <https://js.tensorflow.org/api/latest/>

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

  if (tfvis) tfvis.show.valuesDistribution({ name: 'Data Distribution', tab: 'Visor' }, outputT);
  // check normalization
  // const t = outputT.dataSync();
  // console.log(Math.min(...t), Math.max(...t));

  // model definition
  model.add(tf.layers.dense({ // https://js.tensorflow.org/api/latest/#layers.dense
    name: 'layerDenseInput',
    batchInputShape: [1, params.inputWindow],
    trainable: false,
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
    targetShape: [params.features, Math.trunc(params.neurons / params.features)],
  }));

  const cells = [];
  for (let index = 0; index < params.layers; index++) {
    cells.push(tf.layers[params.cells]({ // https://js.tensorflow.org/api/latest/#Layers-Recurrent
      name: `cell${params.cells}${index}`,
      trainable: true,
      dtype: params.dtype || 'float32',
      recurrentActivation: params.recurrentActivation || 'hardSigmoid',
      units: params.neurons,

      activation: params.activation || 'tanh',
      kernelInitializer: params.kernelInitializer || 'glorotNormal',
      kernelConstraint: params.constraint,
      recurrentInitializer: params.kernelInitializer || 'glorotNormal',
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

  const batchLogs = [];
  function visorPlot(batch, logs) {
    if (!params.visor) return;
    batchLogs.push(logs);
    const values = batchLogs.map((log) => ({ x: log.batch, y: log.loss }));
    tfvis.render.linechart({ name: 'Batch Loss', tab: 'Visor' }, { values, series: ['loss'] });
  }

  // used by fit callback
  function fitEpochCallback(epoch, logs) {
    // if (epoch === 2) model.stopTraining = true;
    const loss = Math.trunc(1000 * Math.sqrt(logs.loss) / (params.dtype === 'int32' ? int : 1)) / 1000;
    if (loss < params.targetLoss) {
      model.stopTraining = true;
      callback(epoch, loss, `Fit early stop: ${epoch} ${loss}`);
    } else {
      callback(epoch, loss);
    }
  }

  // execute fit with callback
  const stats = await model.fit(inputT, outputT, // https://js.tensorflow.org/api/latest/#tf.LayersModel.fit
    { batchSize: params.inputWindow,
      epochs: params.epochs,
      validationSplit: params.validationSplit,
      shuffle: params.shuffle || false,
      // callbacks: tfvis.show.fitCallbacks({ name: 'Training Chart', tab: 'Visor' }, ['loss', 'acc']),
      // callbacks: tf.callbacks.earlyStopping({monitor: 'val_acc'})
      callbacks: {
        onBatchEnd: (batch, logs) => visorPlot(batch, logs),
        onEpochEnd: (epoch, logs) => fitEpochCallback(epoch, logs),
      },
    });
  stats.params = params;
  stats.max = max;
  stats.min = min;

  const evaluateT = model.evaluate(inputT, outputT, { batchSize: params.inputWindow });
  inputT.dispose();
  outputT.dispose();
  if (evaluateT) {
    stats.eval = Math.trunc(100000 * evaluateT[0].dataSync()[0]) / 1000;
    stats.accuracy = Math.trunc(100000 * evaluateT[1].dataSync()[0]) / 1000;
    evaluateT[0].dispose();
    evaluateT[1].dispose();
  }
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
