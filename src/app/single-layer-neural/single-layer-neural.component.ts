import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
@Component({
  selector: 'app-single-layer-neural',
  standalone: true,
  imports: [],
  templateUrl: './single-layer-neural.component.html',
  styleUrl: './single-layer-neural.component.scss'
})
export class SingleLayerNeuralComponent implements OnInit{
  constructor() {
    
  }
  ngOnInit(): void {
    const model = this.createModel();
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

  }

  createModel() {
    // Create a sequential model
    const model = tf.sequential();
  
    // Add a single input layer
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
  
    // Add an output layer
    model.add(tf.layers.dense({units: 1, useBias: true}));
  
    return model;
  }
  
}
