import { AfterViewInit, Component, ElementRef, ViewChild } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { MnistDataService } from './mnist-data.service';
import { Observable, concatMap, forkJoin, from, fromEvent, of, pipe, switchMap, takeUntil, tap } from 'rxjs';
import { CommonModule, NgClass, NgIf } from '@angular/common';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
@Component({
  selector: 'app-hand-writting',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
  ],
  templateUrl: './hand-writting.component.html',
  styleUrl: './hand-writting.component.scss',
})
export class HandWrittingComponent implements AfterViewInit{
  @ViewChild('drawCanvas', { static: false }) drawCanvas!: ElementRef;
  drawCanvasCtx!: CanvasRenderingContext2D;
  trainIndices: any;
  testIndices: any;
  trainImages: any;
  testImages: any;
  trainLabels: any;
  testLabels: any;

  shuffledTrainIndex = 0;
  shuffledTestIndex = 0;

  imgSrc = '';
  hideUI: boolean = true;
  constructor(private mnistDataService: MnistDataService) {}

  ngOnInit() {
    this.loadmnistData();
  }

  async ontrain(){
    const surface = tfvis
          .visor()
          .surface({ name: 'Input Data Examples', tab: 'Input Data' });

        // Get the examples
        const examples = this.nextTestBatch(20);
        const numExamples = examples.xs.shape[0];

        // Create a canvas element to render each example
        for (let i = 0; i < numExamples; i++) {
          const imageTensor: tf.Tensor2D = tf.tidy(() => {
            // Reshape the image to 28x28 px
            return examples.xs
              .slice([i, 0], [1, examples.xs.shape[1]])
              .reshape([28, 28, 1]);
          });

          const canvas = document.createElement('canvas');
          canvas.width = 28;
          canvas.height = 28;
          (canvas as any).style = 'margin: 4px;';
          await tf.browser.toPixels(imageTensor, canvas);
          surface.drawArea.appendChild(canvas);

          imageTensor.dispose();
        }

        const model = this.getModel();
        tfvis.show.modelSummary(
          { name: 'Model Architecture', tab: 'Model' },
          model
        );

        await this.train(model);

        await this.showAccuracy(model);
        //await this.showConfusion(model);

        this.hideUI = false;
        this.generatedModel = model;
  }

  ngAfterViewInit(){
    const canvasEl: HTMLCanvasElement = this.drawCanvas.nativeElement;
    let ctx = canvasEl.getContext('2d');
    if(canvasEl && ctx){
      this.drawCanvasCtx = ctx;
      this.drawCanvasCtx.fillStyle= 'black';
      this.drawCanvasCtx.fillRect(0,0,196,196);
      const mouseDown$:Observable<any> = fromEvent(this.drawCanvas.nativeElement, 'mousedown');
      const mouseMove$:Observable<any> = fromEvent(this.drawCanvas.nativeElement, 'mousemove');
      const mouseUp$:Observable<any> = fromEvent(this.drawCanvas.nativeElement, 'mouseup');
  
      mouseDown$.pipe(concatMap((down) => mouseMove$.pipe(takeUntil(mouseUp$))));
  
      const mouseDraw$ = mouseDown$.pipe(
        tap((e: MouseEvent) => {
          this.drawCanvasCtx.moveTo(e.offsetX, e.offsetY);
        }),
        concatMap(() => mouseMove$.pipe(takeUntil(mouseUp$)))
      );
  
      mouseDraw$.subscribe((e: MouseEvent) => {
        this.drawCanvasCtx.lineTo(e.offsetX, e.offsetY);
        this.drawCanvasCtx.lineWidth = 10;
        this.drawCanvasCtx.strokeStyle = 'white';
        this.drawCanvasCtx.stroke();
      });
    }
  }

  loadmnistData(){
    this.mnistDataService
      .load()
      .pipe(
        tap(([datasetImages, datasetLabels]) => {
          // Create shuffled indices into the train/test set for when we select a
          // random dataset element for training / validation.
          this.trainIndices = tf.util.createShuffledIndices(
            this.mnistDataService.NUM_TRAIN_ELEMENTS
          );
          this.testIndices = tf.util.createShuffledIndices(
            this.mnistDataService.NUM_TEST_ELEMENTS
          );

          // Slice the the images and labels into train and test sets.
          this.trainImages = datasetImages.slice(
            0,
            this.mnistDataService.IMAGE_SIZE *
              this.mnistDataService.NUM_TRAIN_ELEMENTS
          );
          this.testImages = datasetImages.slice(
            this.mnistDataService.IMAGE_SIZE *
              this.mnistDataService.NUM_TRAIN_ELEMENTS
          );
          this.trainLabels = datasetLabels.slice(
            0,
            this.mnistDataService.NUM_CLASSES *
              this.mnistDataService.NUM_TRAIN_ELEMENTS
          );
          this.testLabels = datasetLabels.slice(
            this.mnistDataService.NUM_CLASSES *
              this.mnistDataService.NUM_TRAIN_ELEMENTS
          );
        })
      )
      .subscribe(async (x) => {
        // Create a container in the visor
        this.loadFile();
      });
  }

  generatedModel: tf.Sequential|null=null;

  nextTrainBatch(batchSize: number) {
    return this.nextBatch(
      batchSize,
      [this.trainImages, this.trainLabels],
      () => {
        this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length;
        return this.trainIndices[this.shuffledTrainIndex];
      }
    );
  }

  nextTestBatch(batchSize: number) {
    return this.nextBatch(
      batchSize,
      [this.testImages, this.testLabels],
      this.getIndex
    );
  }

  getIndex = () => {
    this.shuffledTestIndex =
      (this.shuffledTestIndex + 1) % this.testIndices.length;
    return this.testIndices[this.shuffledTestIndex];
  };

  nextBatch(
    batchSize: number,
    [dataSetImages, dataSetLabels]: any[],
    index: Function
  ) {
    const batchImagesArray = new Float32Array(
      batchSize * this.mnistDataService.IMAGE_SIZE
    );
    const batchLabelsArray = new Uint8Array(
      batchSize * this.mnistDataService.NUM_CLASSES
    );

    for (let i = 0; i < batchSize; i++) {
      const idx = index();

      const image = dataSetImages.slice(
        idx * this.mnistDataService.IMAGE_SIZE,
        idx * this.mnistDataService.IMAGE_SIZE +
          this.mnistDataService.IMAGE_SIZE
      );
      batchImagesArray.set(image, i * this.mnistDataService.IMAGE_SIZE);

      const label = dataSetLabels.slice(
        idx * this.mnistDataService.NUM_CLASSES,
        idx * this.mnistDataService.NUM_CLASSES +
          this.mnistDataService.NUM_CLASSES
      );
      batchLabelsArray.set(label, i * this.mnistDataService.NUM_CLASSES);
    }

    const xs = tf.tensor2d(batchImagesArray, [
      batchSize,
      this.mnistDataService.IMAGE_SIZE,
    ]);
    const labels = tf.tensor2d(batchLabelsArray, [
      batchSize,
      this.mnistDataService.NUM_CLASSES,
    ]);

    return { xs, labels };
  }

  getModel(): tf.Sequential {
    const model = tf.sequential();

    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    // In the first layer of our convolutional neural network we have
    // to specify the input shape. Then we specify some parameters for
    // the convolution operation that takes place in this layer.
    model.add(
      tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        dtype:'float32'
      })
    );

    // The MaxPooling layer acts as a sort of downsampling using max values
    // in a region instead of averaging.
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // Repeat another conv2d + maxPooling stack.
    // Note that we have more filters in the convolution.
    model.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
      })
    );
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    model.add(tf.layers.flatten());

    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    const NUM_OUTPUT_CLASSES = 10;
    model.add(
      tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax',

      })
    );

    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    const optimizer = tf.train.adam();
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    return model;
  }

  async train(model: tf.Sequential) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training',
      tab: 'Model',
      styles: { height: '1000px' },
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;

    const [trainXs, trainYs] = tf.tidy(() => {
      const d = this.nextTrainBatch(TRAIN_DATA_SIZE);
      return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
    });

    const [testXs, testYs] = tf.tidy(() => {
      const d = this.nextTestBatch(TEST_DATA_SIZE);
      return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
    });

    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 10,
      shuffle: true,
      callbacks: fitCallbacks,
    });
  }

  readonly classNames = [
    'Zero',
    'One',
    'Two',
    'Three',
    'Four',
    'Five',
    'Six',
    'Seven',
    'Eight',
    'Nine',
  ];

  doPrediction(model: tf.Sequential, testDataSize = 1) {
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = this.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([
      testDataSize,
      IMAGE_WIDTH,
      IMAGE_HEIGHT,
      1,
    ]);
    const labels = testData.labels.argMax(-1);
    const preds = (model.predict(testxs) as tf.Tensor<tf.Rank>).argMax(-1);

    testxs.dispose();
    return [preds, labels];
  }

  async showAccuracy(model: tf.Sequential) {
    const [preds, labels] = this.doPrediction(model);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(
      labels as tf.Tensor1D,
      preds as tf.Tensor1D
    );
    const container = { name: 'Accuracy', tab: 'Evaluation' };
    tfvis.show.perClassAccuracy(container, classAccuracy, this.classNames);

    labels.dispose();
  }

  async showConfusion(model: tf.Sequential) {
    const [preds, labels] = this.doPrediction(model);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(
      labels as tf.Tensor1D,
      preds as tf.Tensor1D
    );
    const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
    tfvis.render.confusionMatrix(container, {
      values: confusionMatrix,
      tickLabels: this.classNames,
    });

    labels.dispose();
  }

  refresh(){
    this.drawCanvasCtx.clearRect(0, 0, this.drawCanvasCtx.canvas.width,  this.drawCanvasCtx.canvas.height);
    this.drawCanvasCtx.beginPath();
    this.drawCanvasCtx.fillStyle= 'black';
    this.drawCanvasCtx.fillRect(0,0, this.drawCanvasCtx.canvas.width,  this.drawCanvasCtx.canvas.height);
  }

  getDrawnImage():Promise<any>{
    const dataUrl = this.drawCanvas.nativeElement.toDataURL();
    let p = new Promise<tf.Tensor<tf.Rank>>((resolve, reject) => {
      var target = new Image();
      target.onload = ()=>{
        
        let image_decode = tf.browser.fromPixels(target, 3);
        const smallImg = tf.image.resizeBilinear(image_decode, [28, 28]);
        const gscale = tf.image.rgbToGrayscale(smallImg);
        let gg:tf.Tensor<tf.Rank> = gscale.reshape([1, 784]);
        resolve(gg);
      }; 
      target.src = dataUrl;
    });
    return p;
  }

  async getTestData(){
    let d:{xs: tf.Tensor2D,labels: tf.Tensor2D} = this.nextTestBatch(1);
    const drawImage = await this.getDrawnImage();
    d.xs = drawImage;
    return d;
  }

  async makePrediction(testDataSize = 1){
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    let testData = await this.getTestData();
    const testxs = testData.xs.reshape([
      testDataSize,
      IMAGE_WIDTH,
      IMAGE_HEIGHT,
      1,
    ]);
    const labels = testData.labels.argMax(-1);
    const predict = this.generatedModel?.predict(testxs);
    console.log('**************predict*********************');
    (predict as any).print();
    console.log((predict as tf.Tensor).dataSync());
    var d = (predict as tf.Tensor).dataSync();
    {
      this.predictedVal = []
      
      d.forEach((x, i: number) => {
          console.log(x);
          this.predictedVal.push({
            i,
            confidence: x.toString().includes('e')?(x * 100).toPrecision(4): (x* 100)
          });
      });
    }
    {
      //check has multiple predictions

    }
    console.log(Math.max(...d));
    //this.confidence = (Math.max(...d) * 100);
    //(predict as tf.Tensor).max(1).mul(100).print(); 

    console.log('******************************************');
    const preds = (predict as tf.Tensor<tf.Rank>).argMax(-1);

    testxs.dispose();
    return [preds, labels];
  }
  confidence:any = '';
  finalPrediction: any;
  predictedVal:any = [];
  async test(){
    this.predictedVal = [];
    if(this.generatedModel){
      const [preds, labels] = await this.makePrediction();
      console.log('**************preds*********************', preds, labels);
      preds.data().then(d=>{
        //this.predictedVal = d
        this.finalPrediction = d[0];
      });
      preds.print();
      console.log('****************************************');
      // const classAccuracy = await tfvis.metrics.perClassAccuracy(
      //   labels as tf.Tensor1D,
      //   preds as tf.Tensor1D
      // );
      // const container = { name: 'Accuracy', tab: 'Evaluation' };
      // tfvis.show.perClassAccuracy(container, classAccuracy, this.classNames);
  
      //labels.dispose();
    }
  }

  modelStr = '';
  serializeModel(){
    //this.generatedModel?.save('localstorage://model-weights')
    this.generatedModel?.save('downloads://model-weights')
    //const weights = this.generatedModel?.getWeights();
    //const j = this.generatedModel?.toJSON();
    //this.modelStr = JSON.stringify(j);
  }

  loadStr(){

    const t = tf.loadLayersModel('localstorage://model-weights')
    //this.generatedModel = t;
    //const j = JSON.parse(this.modelStr);

    t.then(layersModel=>{
      const sequentialModel = tf.sequential();
      layersModel.layers.forEach((layer) => sequentialModel.add(layer));
      this.generatedModel = sequentialModel;
    });

    //const m = this.getModel();
    //m.loadWeights()
  }

  loadFile(){
    //forkJoin([this.mnistDataService.getModel(), this.mnistDataService.getModelWeightBin()])
    //.subscribe(([a,b])=>{
      const t = tf.loadLayersModel('./assets/hand-writting-asset/model-weights.json');
      t.then(layersModel=>{
        console.log(layersModel);
        const sequentialModel = tf.sequential();
          layersModel.layers.forEach((layer) => sequentialModel.add(layer));
          this.generatedModel = sequentialModel;
          this.hideUI = false;
      })
    //})
  }
}
