# LSTM: Time series analysis using machine learning long-short-term-memory

## ToDo

Should use relative data within each inputWindow instead absolute/normalized

## Workflow

### Prepare Data

0. Fetch data
    - Can use any `range` and `interval`, but resulting dataset should be large enough to run analysis on

1. Take set of input data and create and for each point:
    - History: set of size `inputWindow` leading to the point  
      Size is directly proportional to time window sensitivity  
      Idealy should run for multiple sizes
    - Future: set of size `outputWindow` from that point onwards  
      Typically set to 1 to match input to single value, but can be larger

2. Normalize inputs and outputs:
    - For `float32`, values are normalized to 0..1 range  
      This way model:
      - Has room for larger future values
      - Is not sensitive to size of absolute values
      - If `sub` is non-zero (default: 0), normalization is to -sub..+sub  
        so it can take advantage of convergence to zero loss from both sides
    - For `int32`, values are normalized to 0..int *(default: 255)* range and casted to `int32`

### Train Model

3. Fit each input to size of `neurons`:  
    Weights are initialized using `kernelInitializer` algorithm *(default: leCunNormal)*  
    Small values will cause data loss, large values will create more complex matrix  
    Typically a small multiplier of `inputWindow`  

4. Reshape each input to matrix `features` x `neurons / features`  

5. Create cells consisting of `layers` number of LSTM operations that try to match input to output  
   Weights are initialized using `kernelInitializer` algorithm *(default: leCunNormal)*  
   Activate each step using `recurrentActivation` algorithm *(default: relu)*  
   Active output using `activation` algorithm *(default: tanh)*  

6. Connect all cells from (4) using *Recurrent Neural Network*

7. Fit input values to output value(s)
    - Using `inputWindow` as batch size
    - Using `adam` optimizer with `learningRate`
    - Measure loss using `loss` *(default: meanSquaredError)*  
      Loss values are also normalized for consistent return values
    - Set aside `validationSplit` values for validation instead of training

8. Repeat everything `epochs` numbers until *loss* converges to low value
    - Randomize of batches if `shuffle`

9. Evaluate final model using values set aside in `validationSplit` and measure fit
    - Should result in zero or near-zero

### Validate Model

10. Validation model using all known input and output values and measure fit
    - Result is averaged euclidean distance of each predicted output vs real output value

### Future Predictions

11. Predict `predictWindow` future values
    - Predict returns single value, but it's moved forward by using each predicted value as an additional input
