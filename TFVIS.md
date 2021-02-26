cd tfjs-vis
yarn install
yarn upgrade
yarn add esbuild
yarn upgrade -L @tensorflow/tfjs-core @tensorflow/tfjs-layers @tensorflow/tfjs-backend-webgl
node_modules/.bin/esbuild --format=esm --platform=browser --target=esnext --external:@tensorflow --define:process.env.NODE_ENV=\"production\" --bundle src/index.ts --outfile=dist/tfjs-vis.esm.js
