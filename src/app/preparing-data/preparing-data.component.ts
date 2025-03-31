import { Component } from '@angular/core';
import { ICar } from '../models/car.model';
import { DataService } from '../data.service';
import { map } from 'rxjs';
import * as tf from '@tensorflow/tfjs';
import { IMpg_Horsepower } from '../models/impg_horsepower.model';
import { HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-preparing-data',
  standalone: true,
  imports: [HttpClientModule],
  providers: [DataService],
  templateUrl: './preparing-data.component.html',
  styleUrl: './preparing-data.component.scss',
})
export class PreparingDataComponent {
  constructor(private dataService: DataService) {}

  ngOnInit(): void {
    this.dataService
      .getCars()
      .pipe(
        map((cars:ICar[]) => this._mapToMpg_Horsepower(cars)),
      )
      .subscribe((data:IMpg_Horsepower[]) => {
        console.log(this.convertToTensor(data))
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
  convertToTensor(data:IMpg_Horsepower[]) {
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
  
}
