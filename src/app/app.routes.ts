import { Routes } from '@angular/router';
import { TrainTheModelComponent } from './train-the-model/train-the-model.component';
import { HandWrittingComponent } from './hand-writting/hand-writting.component';

export const routes: Routes = [
    {
        path:'2d', component: TrainTheModelComponent,
    },
    {
        path:'handwritting', component: HandWrittingComponent
    },
    {
        path:'', redirectTo:'handwritting', pathMatch:'full'
    },
    {
        path:'**', redirectTo:'handwritting'
    }
];
