#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <time.h> 
 
int main(int argc, char* argv[]) { 
     
    double time_spent1 = 0.0; 
 
    clock_t begin1 = clock();  
     
    // Check if enough arguments are provided 
    if (argc < 4) { 
        printf("Usage: ./program_name Matrix accuracy iterations\n"); 
        return 1; 
    } 
 
    // Convert command line arguments to integers 
    int Matrix = atoi(argv[1]); 
    double accuracy = atof(argv[2]); 
    int iterations = atoi(argv[3]); 
     
    // Allocate 2D arrays on host memory 
    double* arr = (double*)malloc(Matrix * Matrix * sizeof(double)); 
    double* array_new = (double*)malloc(Matrix * Matrix * sizeof(double)); 
    // Initialize arrays to zero 
    for (int i = 0; i < Matrix * Matrix; i++) { 
        arr[i] = 0; 
        array_new[i] = 0; 
    } 
    // Set boundary conditions 
    arr[0 * Matrix + 0] = 10; 
    arr[0 * Matrix + Matrix - 1] = 20; 
    arr[(Matrix - 1) * Matrix + 0] = 30; 
    arr[(Matrix - 1) * Matrix + Matrix - 1] = 20; 
     
    for (int j = 1; j < Matrix; j++) { 
            arr[0 * Matrix + j] = (arr[0 * Matrix + Matrix - 1] - arr[0 * Matrix + 0]) / (Matrix - 1) + arr[0 * Matrix + j - 1];   //top 
            arr[(Matrix - 1) * Matrix + j] = (arr[(Matrix - 1) * Matrix + Matrix - 1] - arr[(Matrix - 1) * Matrix + 0]) / (Matrix - 1) + arr[(Matrix - 1) * Matrix + j - 1]; //bottom 
            arr[j * Matrix + 0] = (arr[(Matrix - 1) * Matrix + 0] - arr[0 * Matrix + 0]) / (Matrix - 1) + arr[(j - 1) * Matrix + 0]; //left 
            arr[j * Matrix + Matrix - 1] = (arr[(Matrix - 1) * Matrix + Matrix - 1] - arr[0 * Matrix + Matrix - 1]) / (Matrix - 1) + arr[(j - 1) * Matrix + Matrix - 1]; //right 
        } 
    // Main loop 
    double err = accuracy + 1; 
    int iter = 0; 
 
#pragma acc data copy(arr[0:Matrix*Matrix], array_new[0:Matrix*Matrix]) 
    { 
        while (err > accuracy  && iter < iterations) { 
            // Compute new values 
            err = 0; 
#pragma acc parallel reduction(max:err) 
{ 
    #pragma acc loop independent 
            for (int j = 1; j < Matrix - 1; j++) { 
#pragma acc loop independent 
                for (int i = 1; i < Matrix - 1; i++) { 
                    int index = j * Matrix + i; 
                    array_new[index] = 0.25 * (arr[index + Matrix] + arr[index - Matrix] + 
                        arr[index - 1] + arr[index + 1]); 
                    err = fmax(err, fabs(array_new[index] - arr[index])); 
                } 
            } 
} 
            // Update values 
#pragma acc kernels loop independent 
            for (int j = 1; j < Matrix - 1; j++) { 
#pragma acc loop 
                for (int i = 1; i < Matrix - 1; i++) { 
                    int index = j * Matrix + i; 
                    arr[index] = array_new[index]; 
                } 
            } 
 
            iter++; 
        } 
    } 
 
    printf("Final result: %d, %0.6lf\n", iter, err); 
    
    
     //вывод сетки размером 15*15
        if (Matrix==15){
                for (int i = 0; i <Matrix; i++) {
                    for (int j = 0; j < Matrix; j++) {
                        printf("%0.2lf ", array_new[i * Matrix + j]);
                    }   
                printf("\n");
                }
            }
    // Free memory 
    free(arr); 
    free(array_new); 
 
     
    clock_t end1 = clock(); 
    time_spent1 += (double)(end1 - begin1) / CLOCKS_PER_SEC; 
    printf("%f\n", time_spent1); 
     
    return 0; 
}
