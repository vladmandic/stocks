import * as tf from '@tensorflow/tfjs';

async function train(input, output, params, callback) {
  // console.log('Training:', params, input, output);
  const model = tf.sequential();

  // normalize inputs
  const max = Math.max(...output);
  const mul = 255;
  const inputT = params.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(input, [input.length, params.inputWindow]).div(max).mul(mul).toInt())
    : tf.tidy(() => tf.tensor2d(input, [input.length, params.inputWindow]).div(max));
  const outputT = params.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(output, [output.length, params.outputWindow]).div(max).mul(mul).toInt())
    : tf.tidy(() => tf.tensor2d(output, [output.length, params.outputWindow]).div(max));

  // model definition
  model.add(tf.layers.dense({ units: params.neurons, inputShape: [params.inputWindow] }));
  model.add(tf.layers.reshape({ targetShape: [params.features, Math.trunc(params.neurons / params.features)] }));
  const cell = [];
  for (let index = 0; index < params.layers; index++) {
    cell.push(tf.layers.lstmCell({
      // https://js.tensorflow.org/api/latest/#layers.lstmCell
      units: params.inputWindow,
      dtype: params.dtype,
    }));
  }
  model.add(tf.layers.rnn({
    // https://js.tensorflow.org/api/latest/#layers.rnn
    cell,
    inputShape: [params.features, Math.trunc(params.neurons / params.features)],
    returnSequences: false,
    dtype: params.dtype,
  }));
  model.add(tf.layers.dense({
    // https://js.tensorflow.org/api/latest/#layers.dense
    units: params.outputWindow,
    inputShape: [params.inputWindow],
    dtype: params.dtype,
  }));
  // compile model
  model.compile({
    // https://js.tensorflow.org/api/latest/#train.adam
    optimizer: tf.train.adam(params.learningRate),
    loss: 'meanSquaredError',
  });
  // execute fit with callback
  // https://js.tensorflow.org/api/latest/#tf.LayersModel.fit
  const stats = await model.fit(inputT, outputT,
    { batchSize: params.inputWindow,
      epochs: params.epochs,
      callbacks: {
        onEpochEnd: (epoch, logs) => callback(epoch, logs),
      },
    });
  stats.factor = max;
  stats.multiplier = mul;
  stats.dtype = params.dtype;
  inputT.dispose();
  outputT.dispose();
  return { model, stats };
}

async function predict(X, model) {
  const inputT = model.stats.dtype === 'int32'
    ? tf.tidy(() => tf.tensor2d(X, [X.length, X[0].length]).div(model.stats.factor).mul(model.stats.multiplier).toInt())
    : tf.tidy(() => tf.tensor2d(X, [X.length, X[0].length]).div(model.stats.factor));
  const outputT = model.model.predict(inputT);
  const output = outputT.dataSync().map((val) => val * model.stats.factor);
  inputT.dispose();
  outputT.dispose();
  return output;
}

export { train, predict };
