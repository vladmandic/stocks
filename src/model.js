import * as tf from '@tensorflow/tfjs';

async function train(input, output, params, callback) {
  // console.log('Training:', params, input, output);
  const model = tf.sequential();

  const max = Math.max(...output);
  const mul = 255;
  const inputT = params.dtype === 'int'
    ? tf.tidy(() => tf.tensor2d(input, [input.length, params.windowSize]).div(max).mul(mul).toInt())
    : tf.tidy(() => tf.tensor2d(input, [input.length, params.windowSize]).div(max));
  const outputT = params.dtype === 'int'
    ? tf.tidy(() => tf.tensor2d(output, [output.length, 1]).div(max).mul(mul).toInt())
    : tf.tidy(() => tf.tensor2d(output, [output.length, 1]).div(max));

  // model definition
  model.add(tf.layers.dense({ units: params.neurons, inputShape: [params.windowSize] }));
  model.add(tf.layers.reshape({ targetShape: [params.features, Math.trunc(params.neurons / params.features)] }));
  const cell = [];
  for (let index = 0; index < params.layers; index++) {
    cell.push(tf.layers.lstmCell({
      units: params.windowSize,
    }));
  }
  model.add(tf.layers.rnn({
    cell,
    inputShape: [params.features, Math.trunc(params.neurons / params.features)],
    returnSequences: false,
  }));
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [params.windowSize],
  }));
  // compile model
  model.compile({
    optimizer: tf.train.adam(params.learningRate),
    loss: params.loss,
  });
  // execute fit with callback
  const stats = await model.fit(inputT, outputT,
    { batchSize: params.windowSize,
      epochs: params.epochs,
      callbacks: { onEpochEnd: (epoch, logs) => callback(epoch, logs) },
    });
  stats.factor = max;
  stats.multiplier = mul;
  stats.dtype = params.dtype;
  inputT.dispose();
  outputT.dispose();
  return { model, stats };
}

async function predict(X, model) {
  const inputT = model.stats.dtype === 'int'
    ? tf.tidy(() => tf.tensor2d(X, [X.length, X[0].length]).div(model.stats.factor).mul(model.stats.multiplier).toInt())
    : tf.tidy(() => tf.tensor2d(X, [X.length, X[0].length]).div(model.stats.factor));
  const outputT = model.model.predict(inputT);
  const output = outputT.dataSync().map((val) => val * model.stats.factor);
  inputT.dispose();
  outputT.dispose();
  return output[0];
}

export { train, predict };
