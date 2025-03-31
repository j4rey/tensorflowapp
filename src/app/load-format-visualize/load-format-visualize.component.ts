import { HttpClientModule } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import * as tfvis from '@tensorflow/tfjs-vis';
import { DataService } from '../data.service';
import { map } from 'rxjs';
import { ICar } from '../models/car.model';
import { IMpg_Horsepower } from '../models/impg_horsepower.model';

@Component({
  selector: 'app-load-format-visualize',
  standalone: true,
  imports: [HttpClientModule],
  templateUrl: './load-format-visualize.component.html',
  styleUrl: './load-format-visualize.component.scss',
  providers: [DataService],
})
export class LoadFormatVisualizeComponent implements OnInit {
  constructor(private dataService: DataService) {}

  ngOnInit(): void {
    this.dataService
      .getCars()
      .pipe(
        map((cars:ICar[]) => this._mapToMpg_Horsepower(cars)),
        map((cars:IMpg_Horsepower[])=> this._mapToVisualize(cars))
      )
      .subscribe((values:any[]) => {
        tfvis.render.scatterplot(
          {name: 'Horsepower v MPG'},
          {values},
          {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
          }
        )
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

  private _mapToVisualize(cars: IMpg_Horsepower[]): any {
    return cars.map((car: IMpg_Horsepower)=>({
        x: car.horsepower,
        y: car.mpg
      }));
  }
}