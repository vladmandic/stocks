import * as tf from '@tensorflow/tfjs';

function str(...msg) {
  if (!Array.isArray(msg)) return msg;
  let line = '';
  for (const entry of msg) {
    if (typeof entry === 'object') line += JSON.stringify(entry).replace(/{|}|"|\[|\]/g, '').replace(/,/g, ' ');
    else line += entry;
  }
  return line;
}

function log(...msg) {
  const dt = new Date();
  const ts = `${dt.getHours().toString().padStart(2, '0')}:${dt.getMinutes().toString().padStart(2, '0')}:${dt.getSeconds().toString().padStart(2, '0')}.${dt.getMilliseconds().toString().padStart(3, '0')}`;
  // eslint-disable-next-line no-console
  console.log(ts, ...msg);
  document.getElementById('log').innerHTML += `${ts} &nbsp ${str(msg)}<br>`;
}

async function main() {
  log('ltsm initializing');
  await tf.setBackend('webgl');
  await tf.enableProdMode();
  // tf.ENV.set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
  // tf.ENV.set('WEBGL_FORCE_F16_TEXTURES', true);
  // tf.ENV.set('WEBGL_PACK_DEPTHWISECONV', true);
  const gl = await tf.backend().getGPGPUContext().gl;
  log('tfjs version:', tf.version_core, 'backend:', tf.getBackend(), 'flags:', tf.ENV.flags);
  log(`gl version:${gl.getParameter(gl.VERSION)} renderer:${gl.getParameter(gl.RENDERER)}`);
  await tf.ready();
}

window.onload = main;
