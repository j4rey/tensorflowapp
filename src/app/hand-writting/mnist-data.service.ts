import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable, forkJoin, from, map, switchMap, tap } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class MnistDataService {
  readonly IMAGE_SIZE = 784;
  readonly NUM_CLASSES = 10;
  readonly NUM_DATASET_ELEMENTS = 65000;
  
  readonly NUM_TRAIN_ELEMENTS = 55000;
  get NUM_TEST_ELEMENTS(){
    return this.NUM_DATASET_ELEMENTS - this.NUM_TRAIN_ELEMENTS;
  }
  readonly MNIST_IMAGES_SPRITE_PATH =
      './assets/hand-writting-asset/mnist_images.png';
      readonly MNIST_LABELS_PATH =
      './assets/hand-writting-asset/mnist_labels_uint8';


  constructor(private httpClient: HttpClient) { }

  load() {
    return forkJoin([this.getDatasetImages(), this.getDatasetLabels()]) 
  }

  getDatasetImages(): Observable<any>{
    return this.httpClient.get(this.MNIST_IMAGES_SPRITE_PATH,{responseType:'blob'})
    .pipe(
      switchMap(x=>{
        let p = new Promise((resolve, reject) => {
          const img = new Image();
  
          img.onload = () => {
            resolve({
              img: img,
              width: img.width,
              height: img.height
            });
          };
  
          img.onerror = (error) => {
            reject(error);
          };
  
          img.src = URL.createObjectURL(x);
        });
        return from(p);
      }),
      map(({img, width, height}:any)=>{
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const datasetBytesBuffer =
            new ArrayBuffer(this.NUM_DATASET_ELEMENTS * this.IMAGE_SIZE * 4);
        
        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < this.NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
              datasetBytesBuffer, i * this.IMAGE_SIZE * chunkSize * 4,
              this.IMAGE_SIZE * chunkSize);
          ctx?.drawImage(
              img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
              chunkSize);

          const imageData = ctx?.getImageData(0, 0, canvas.width, canvas.height);

          if(imageData)
          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        let datasetImages = new Float32Array(datasetBytesBuffer);
        return datasetImages;
      })
    );
  }

  getDatasetLabels(){
    return this.httpClient.get(this.MNIST_LABELS_PATH,{responseType:'blob', observe: 'response'})
    .pipe(
      switchMap(x=>{
        if(x.body)
        return from(x?.body?.arrayBuffer());
      else
        throw 'failed to get dataset labels'
      }),
      map((labelsResponseArrayBuffer)=>{
        const datasetLabels = new Uint8Array(labelsResponseArrayBuffer);
        return datasetLabels;
      })
    );
    
  }


  getModel(){
    return this.httpClient.get('./assets/hand-writting-asset/model-weights.json',{responseType:'blob', observe: 'response'})
  }

  getModelWeightBin(){
    return this.httpClient.get('./assets/hand-writting-asset/model-weights.weights.bin',{responseType:'blob', observe: 'response'})
  }
}
