import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLinkWithHref, RouterModule, RouterOutlet } from '@angular/router';
import { SequentialPredictComponent } from './sequential-predict/sequential-predict.component';
import { LoadFormatVisualizeComponent } from './load-format-visualize/load-format-visualize.component';
import { SingleLayerNeuralComponent } from './single-layer-neural/single-layer-neural.component';
import { PreparingDataComponent } from './preparing-data/preparing-data.component';
import { TrainTheModelComponent } from './train-the-model/train-the-model.component';
import { routes } from './app.routes';
import { HandWrittingComponent } from './hand-writting/hand-writting.component';

const components = [
  SequentialPredictComponent,
  LoadFormatVisualizeComponent,
  SingleLayerNeuralComponent,
  PreparingDataComponent,
  TrainTheModelComponent
]

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule, 
    RouterOutlet,
    RouterLinkWithHref,
    HandWrittingComponent,
  ],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
})
export class AppComponent {
  title = 'tensorflowapp';

  ngOnInit() {
    
  }
}
