Training | backend:webgl | dtype:float32 | evalError:2.5 | smaError:2.5 | inputWindow:30 | outputWindow:1 | predictWindow:60 | epochs:25 | validationSplit:0.2 | optimizer:adam | learningRate:0.002 | loss:meanSquaredError | neurons:40 | features:10 | layers:1 | cells:lstmCell | kernelInitializer:leCunNormal | activation:relu | recurrentActivation:hardSigmoid | forgetBias:false | biasInitializer:glorotNormal | shuffle:false
OK | Training loss: 0.076
OK | Model evaluation: 0.616% error
OK | Model fit RMS: 3.09% | SMA RMS: 6.62%

Training | backend:webgl | dtype:float32 | evalError:2.5 | smaError:2.5 | inputWindow:30 | outputWindow:1 | predictWindow:60 | epochs:25 | validationSplit:0.2 | optimizer:adamax | learningRate:0.002 | loss:meanSquaredError | neurons:40 | features:10 | layers:1 | cells:gruCell | kernelInitializer:glorotNormal | activation:linear | recurrentActivation:sigmoid | forgetBias:false | biasInitializer:glorotNormal | shuffle:false
OK | Training loss: 0.085
OK | Model evaluation: 0.711% error
OK | Model fit RMS: 3.32% | SMA RMS: 6.62%
OK | Predict correction to SMA: -1.09%

Training | backend:webgl | dtype:float32 | evalError:2.5 | smaError:2.5 | inputWindow:30 | outputWindow:1 | predictWindow:60 | epochs:25 | validationSplit:0.2 | optimizer:adamax | learningRate:0.02 | loss:meanSquaredError | neurons:40 | features:10 | layers:1 | cells:gruCell | kernelInitializer:heNormal | activation:softmax | recurrentActivation:softmax | forgetBias:false | biasInitializer:heNormal | shuffle:false
OK | Training loss: 0.067
OK | Model evaluation: 0.442% error
OK | Model fit RMS: 2.62% | SMA RMS: 6.62%
OK | Predict correction to SMA: -3.34%
