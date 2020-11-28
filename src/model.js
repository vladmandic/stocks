import * as tf from '@tensorflow/tfjs';

async function train(input, output, params, callback) {
  // console.log('Training:', params, input, output);
  const model = tf.sequential();

  // normalize inputs
  const max = Math.max(...output);
  const inputT = params.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(input, [input.length, params.inputWindow]).mul(255).div(max).toInt())
    : tf.tidy(() => tf.tensor2d(input, [input.length, params.inputWindow]).div(max));
  const outputT = params.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(output, [output.length, params.outputWindow]).mul(255).div(max).toInt())
    : tf.tidy(() => tf.tensor2d(output, [output.length, params.outputWindow]).div(max));

  // model definition
  model.add(tf.layers.dense({ // https://js.tensorflow.org/api/latest/#layers.dense
    name: 'layerDenseInput',
    units: params.neurons,
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
      unitForgetBias: true,
      recurrentActivation: 'hardSigmoid',
      activation: 'tanh',
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
    dtype: params.dtype,
  }));

  // compile model
  model.compile({ // https://js.tensorflow.org/api/latest/#train.adam
    optimizer: tf.train.adam(params.learningRate),
    loss: 'meanSquaredError',
  });

  // used by fit callback
  function normalizeLoss(epoch, logs) {
    const loss = Math.trunc(1000 * Math.sqrt(logs.loss) / (params.dtype === 'int32' ? 255 : 1)) / 1000;
    callback(epoch, loss);
  }

  // execute fit with callback
  const stats = await model.fit(inputT, outputT, // https://js.tensorflow.org/api/latest/#tf.LayersModel.fit
    { batchSize: params.inputWindow,
      epochs: params.epochs,
      callbacks: {
        onEpochEnd: (epoch, logs) => normalizeLoss(epoch, logs),
      },
    });
  stats.dtype = params.dtype;
  inputT.dispose();
  outputT.dispose();
  console.log('Model:', model);
  return { model, stats };
}

async function predict(model, input) {
  const max = Math.max(...input);
  const inputT = model.stats.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(input, [1, input.length]).mul(255).div(max).toInt())
    : tf.tidy(() => tf.tensor2d(input, [1, input.length]).div(max));
  const outputT = model.model.predict(inputT);
  const normalizeT = outputT.mul(max);
  const output = normalizeT.dataSync();
  inputT.dispose();
  normalizeT.dispose();
  outputT.dispose();
  return output;
}

export { train, predict };
