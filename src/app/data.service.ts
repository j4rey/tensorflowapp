import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { ICar } from './models/car.model';

@Injectable({
  providedIn: 'root'
})
export class DataService {

  constructor(private httpClient: HttpClient) { }

  getCars(): Observable<ICar[]>{
    return this.httpClient.get<ICar[]>('https://storage.googleapis.com/tfjs-tutorials/carsData.json')
  }
}
