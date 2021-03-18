#!/usr/bin/env -S node --trace-warnings

const esbuild = require('esbuild');
const log = require('@vladmandic/pilogger');

// keeps esbuild service instance cached
const banner = { js: `
  /*
  Time Series Analysis
  homepage: <https://github.com/vladmandic/stocks>
  author: <https://github.com/vladmandic>'
  */
` };

// common configuration
const target = {
  banner,
  minifyWhitespace: false,
  minifyIdentifiers: false,
  minifySyntax: false,
  bundle: true,
  sourcemap: true,
  logLevel: 'error',
  target: 'es2018',
  platform: 'browser',
  format: 'esm',
  metafile: true,
  entryPoints: ['src/index.js'],
  outfile: 'dist/index.js',
  external: ['fs', 'buffer', 'util', 'os'],
};

async function getStats(json) {
  const stats = {};
  if (json && json.metafile?.inputs && json.metafile?.outputs) {
    for (const [key, val] of Object.entries(json.metafile.inputs)) {
      if (key.startsWith('node_modules')) {
        stats.modules = (stats.modules || 0) + 1;
        stats.moduleBytes = (stats.moduleBytes || 0) + val.bytes;
      } else {
        stats.imports = (stats.imports || 0) + 1;
        stats.importBytes = (stats.importBytes || 0) + val.bytes;
      }
    }
    const files = [];
    for (const [key, val] of Object.entries(json.metafile.outputs)) {
      if (!key.endsWith('.map')) {
        files.push(key);
        stats.outputBytes = (stats.outputBytes || 0) + val.bytes;
      }
    }
    stats.outputFiles = files.join(', ');
  }
  return stats;
}

// rebuild on file change
async function build(f, msg) {
  log.info('Build: file', msg, f);
  try {
    // @ts-ignore
    const meta = await esbuild.build(target);
    const stats = await getStats(meta);
    log.state('Build stats:', stats);
  } catch (err) {
    log.error('Build error', JSON.stringify(err.errors || err, null, 2));
    if (require.main === module) process.exit(1);
  }
}

if (require.main === module) {
  log.header();
  build('all', 'startup');
} else {
  exports.build = build;
}
