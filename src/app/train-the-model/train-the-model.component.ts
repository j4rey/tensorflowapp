import { HttpClientModule } from '@angular/common/http';
import { Component } from '@angular/core';
import { DataService } from '../data.service';

import { from, map, switchMap } from 'rxjs';
import { ICar } from '../models/car.model';
import { IMpg_Horsepower } from '../models/impg_horsepower.model';
import { ModelFitArgs } from '@tensorflow/tfjs';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';



/*
The steps in training a machine learning model include:

Formulate your task:

    Is it a regression problem or a classification one?
    Can this be done with supervised learning or unsupervised learning?
    What is the shape of the input data? What should the output data look like?

Prepare your data:

    Clean your data and manually inspect it for patterns when possible
    Shuffle your data before using it for training
    Normalize your data into a reasonable range for the neural network. Usually 0-1 or -1-1 are good ranges for numerical data.
    Convert your data into tensors

Build and run your model:

    Define your model using tf.sequential or tf.model then add layers to it using tf.layers.*
    Choose an optimizer ( adam is usually a good one), and parameters like batch size and number of epochs.
    Choose an appropriate loss function for your problem, and an accuracy metric to help your evaluate progress. meanSquaredError is a common loss function for regression problems.
    Monitor training to see whether the loss is going down

Evaluate your model

    Choose an evaluation metric for your model that you can monitor while training. Once it's trained, try making some test predictions to get a sense of prediction quality.

*/
@Component({
  selector: 'app-train-the-model',
  standalone: true,
  imports: [HttpClientModule],
  providers: [DataService],
  templateUrl: './train-the-model.component.html',
  styleUrl: './train-the-model.component.scss'
})
export class TrainTheModelComponent {
  constructor(private dataService: DataService) {}

  ngOnInit(): void {
    this.dataService
      .getCars()
      .pipe(
        map((cars:ICar[]) => this._mapToMpg_Horsepower(cars)),
        switchMap((data:IMpg_Horsepower[])=>{
          // Create sequential model
          const model = this._createModel();

          // Convert the data to a form we can use for training.
          const tensorData = this._convertToTensor(data);
          const {inputs, labels} = tensorData;
          // Train the model
          const trainingPromise = this._trainModel(model, inputs, labels);
          return from(trainingPromise)
                .pipe(
                  map(history=>({
                    history,
                    model,
                    data,
                    tensorData
                  }))
                );
        })
      )
      .subscribe(({history, model, data, tensorData}) => {
        console.log('Done Training');
        //console.log(history);
        this._testModel(model, data, tensorData);
      });
  }
  private _mapToMpg_Horsepower(cars:ICar[]){
    return cars.map((car:ICar)=> {
      return {
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
      } as IMpg_Horsepower;
    })
    .filter((car:IMpg_Horsepower)=> (car.mpg != null && car.horsepower != null))
  }

 

  /**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
  _convertToTensor(data:IMpg_Horsepower[]) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.
  
    return tf.tidy(() => {
      // Step 1. Shuffle the data
      tf.util.shuffle(data);
  
      // Step 2. Convert data to Tensor
      const inputs = data.map(d => d.horsepower)
      const labels = data.map(d => d.mpg); //outputs
  
      //The tensor will have a shape of [num_examples, num_features_per_example]. 
      //Here we have inputs.length examples and each example has 1 input feature (the horsepower).
      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
  
      //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();
  
      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
  
      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      }
    });
  }

  _createModel():tf.Sequential {
    // Create a sequential model
    const model = tf.sequential();
  
    // Add a single input layer
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
    model.add(tf.layers.dense({units: 25, activation: 'sigmoid'}));


    // Add an output layer
    model.add(tf.layers.dense({units: 1, useBias: true, activation: 'sigmoid'}));
  
    return model;
  }
  _trainModel(model: tf.Sequential, inputs: tf.Tensor<tf.Rank>, labels: tf.Tensor<tf.Rank>): Promise<tf.History> {
    // Prepare the model for training.
    model.compile({
      //This is the algorithm that is going to govern the updates to the model as it sees examples. 
      //There are many optimizers available in TensorFlow.js.
      //Here we have picked the adam optimizer as it is quite effective in practice and requires no configuration.
      optimizer: tf.train.adam(),
      //This is a function that will tell the model how well it is doing on learning each of the batches (data subsets) that it is shown. 
      //Here we use meanSquaredError to compare the predictions made by the model with the true values.
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    });
  
    //batchSize refers to the size of the data subsets that the model will see on each iteration of training.
    //Common batch sizes tend to be in the range 32-512. There isn't really an ideal batch size for all problems.
    const batchSize = 32;
    //epochs refers to the number of times the model is going to look at the entire dataset that you provide it.
    const epochs = 500;
  
    //start the training loop
    return model.fit(inputs, labels, 
      
      {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'],
        { height: 200, callbacks: ['onEpochEnd'] }
      )
    } as ModelFitArgs);
  }
  
  _testModel(model: tf.Sequential, inputData:IMpg_Horsepower[], normalizationData:{
    inputs:tf.Tensor<tf.Rank>, 
    labels:tf.Tensor<tf.Rank>,
    inputMax: tf.Tensor<tf.Rank>,
    inputMin: tf.Tensor<tf.Rank>,
    labelMin: tf.Tensor<tf.Rank>,
    labelMax: tf.Tensor<tf.Rank>
  }) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;
  
    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {
  
      const xsNorm = tf.linspace(0, 1, 100);
      const predictions: tf.Tensor<tf.Rank> = model.predict(xsNorm.reshape([100, 1])) as tf.Tensor<tf.Rank>;
  
      const unNormXs = xsNorm
        .mul(inputMax.sub(inputMin))
        .add(inputMin);
  
      const unNormPreds = predictions
        .mul(labelMax.sub(labelMin))
        .add(labelMin);
  
      // Un-normalize the data
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });
  
  
    const predictedPoints = Array.from(xs).map((val, i) => {
      return {x: val, y: preds[i]}
    });
  
    const originalPoints = inputData.map(d => ({
      x: d.horsepower, y: d.mpg,
    }));
  
  
    tfvis.render.scatterplot(
      {name: 'Model Predictions vs Original Data'},
      {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
      {
        xLabel: 'Horsepower',
        yLabel: 'MPG',
        height: 300
      }
    );
  }
  
}
