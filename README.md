# Time series analysis using machine learning long-short-term-memory

## Todo

- Multi-parameter model

## Workflow

### Prepare Data

0. Fetch data
    - Can use any `range` and `interval`  
      Resulting dataset should be large enough to run analysis on

1. Take set of input data and create and for each point:
    - History: set of size `inputWindow` leading to the point  
      Size is directly proportional to time window sensitivity  
      Idealy should run for multiple sizes
    - Future: set of size `outputWindow` from that point onwards  
      Typically set to 1 to match input to single value, but can be larger

2. Normalize inputs and outputs:
    - For `float32`, values are normalized to -0.75..+0.75 range  
      This way model:
      - Has room for future values outside of the range
      - Is not sensitive to size of absolute values
      - It can take advantage of convergence to zero loss from both sides
    - For `int32`, values are normalized to 0..255 range and casted to `int32`
    - On prediction, results are de-normalized to original ranges

### Train Model

3. Fit each input to size of `neurons`:  
    Weights are initialized using `kernelInitializer` algorithm  
    Small values will cause data loss, large values will create more complex matrix  
    Typically a small multiplier of `inputWindow`  

4. Reshape each input to matrix `[features]` x `[neurons / features]`  

5. Number of `layers`, each processing `cells` operation that try to match input to output  
   Weights are initialized using `kernelInitializer` algorithm  
   Activate each step using `recurrentActivation` algorithm  
   Active output using `activation` algorithm

6. Connect all cells from (4) using *RNN: Recurrent Neural Network*

7. Fit input values to output value(s)
    - Using `inputWindow` as batch size
    - Using `optimizer` with `learningRate`
    - Measure training `loss`  
      Loss values are also normalized for consistent return values
    - Set aside `validationSplit` values for validation instead of training

8. Repeat everything `epochs` numbers until *loss* converges to low value
    - Randomize of batches if `shuffle`
    - Discard learned bias if `forgetBias` and reinitialize it using `biasInitializer`

9. Evaluate final model using values set aside in `validationSplit` and measure fit
    - Results are discared if above `evalError`

### Validate Model

10. Validation model using all known input and output values and measure fit
    - Results are compared between and discared if above `smaError`:
      - Averaged root-median-square for all predicted outputs
      - Averaged root-median-square for SMA values for the same `inputWindow`

### Future Predictions

11. Predict `predictWindow` future values
    - Predict returns single value and it's moved forward by using each predicted value as an additional input
    - Prediction is discarded if its beyond sensible range based on min/max values of input
