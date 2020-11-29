import * as tf from '@tensorflow/tfjs';

const int = 256;

async function train(input, output, params, callback) {
  // console.log('Training:', params, input, output);
  const model = tf.sequential();

  // normalize inputs
  const max = Math.max(...output);
  const inputT = params.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(input, [input.length, params.inputWindow]).mul(int).div(max).toInt())
    : tf.tidy(() => tf.tensor2d(input, [input.length, params.inputWindow]).div(max));
  const outputT = params.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(output, [output.length, params.outputWindow]).mul(int).div(max).toInt())
    : tf.tidy(() => tf.tensor2d(output, [output.length, params.outputWindow]).div(max));

  // model definition
  model.add(tf.layers.dense({ // https://js.tensorflow.org/api/latest/#layers.dense
    name: 'layerDenseInput',
    units: params.neurons,
    kernelInitializer: params.kernelInitializer,
    inputShape: [params.inputWindow],
  }));

  model.add(tf.layers.reshape({ // https://js.tensorflow.org/api/latest/#layers.reshape
    name: 'layerReshape',
    targetShape: [params.features, Math.trunc(params.neurons / params.features)],
  }));

  const cells = [];
  for (let index = 0; index < params.layers; index++) {
    cells.push(tf.layers.lstmCell({ // https://js.tensorflow.org/api/latest/#layers.lstmCell
      name: `LayerLTSM${index}`,
      units: params.inputWindow,
      dtype: params.dtype,
      unitForgetBias: params.forgetBias,
      kernelInitializer: params.kernelInitializer,
      recurrentActivation: params.recurrentActivation,
      activation: params.activation,
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
    units: params.outputWindow,
    kernelInitializer: params.kernelInitializer,
    dtype: params.dtype,
  }));

  // compile model
  // const rate = params.dtype === 'int32' ? params.learningRate * int : params.learningRate;
  const rate = params.learningRate;
  model.compile({ // https://js.tensorflow.org/api/latest/#train.adam
    optimizer: tf.train.adam(rate),
    loss: params.loss,
  });

  // used by fit callback
  function normalizeLoss(epoch, logs) {
    const loss = Math.trunc(1000 * Math.sqrt(logs.loss) / (params.dtype === 'int32' ? int : 1)) / 1000;
    callback(epoch, loss);
  }

  // execute fit with callback
  const stats = await model.fit(inputT, outputT, // https://js.tensorflow.org/api/latest/#tf.LayersModel.fit
    { batchSize: params.inputWindow,
      epochs: params.epochs,
      validationSplit: params.validationSplit,
      shuffle: params.shuffle,
      callbacks: {
        onEpochEnd: (epoch, logs) => normalizeLoss(epoch, logs),
        // onBatchEnd: (batch, logs) => console.log(batch, logs),
      },
    });
  stats.params = params;

  const evaluateT = model.evaluate(inputT, outputT, { batchSize: params.inputWindow });
  stats.eval = Math.trunc(10000 * evaluateT.dataSync()[0] / int) / 10000;
  inputT.dispose();
  outputT.dispose();
  evaluateT.dispose();
  return { model, stats };
}

async function predict(model, input) {
  const max = Math.max(...input);
  const inputT = model.stats.params.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(input, [1, input.length]).mul(int).div(max).toInt())
    : tf.tidy(() => tf.tensor2d(input, [1, input.length]).div(max));
  const outputT = model.model.predict(inputT, { batchSize: model.stats.params.inputWindow });
  const normalizeT = model.stats.params.dtype === 'int32'
    ? tf.tidy(() => outputT.mul(max).div(int))
    : tf.tidy(() => outputT.mul(max));
  const output = normalizeT.dataSync();
  inputT.dispose();
  normalizeT.dispose();
  outputT.dispose();
  return output;
}

export { train, predict };
