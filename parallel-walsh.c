#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <stdbool.h>
#include <string.h>
// final cloud version by Yosef 021215 2203
// testing, git manipulation from ubuntu terminal
// testing the ubuntu terminal again, 311016 2146

// recursive calculation of FFT:
void recursive_fft(int* vector, int start, int size){
    int half = size / 2;
    // recursive base case: half is 1 and no further change is neccesarry 
    if(half != 1) {
        // if half > 1, split up and do recursive computation
        recursive_fft(vector, start, half);
        recursive_fft(vector, start + half, half);
    }
    
    int* ptr = vector + start;
    // finish up, after changes:
    for(int im = 0; im < half; im++){
        // cell are in first half:
        ptr[im] += ptr[half + im];
        ptr[im + half] = ptr[im] - 2 * ptr[im + half];
    }
}

void fast_parallel_walsh(int* vector, int size)
{
    // first save num_threads
    int num_threads;
    #pragma omp parallel shared(num_threads)
    {
        num_threads = omp_get_num_threads();
    }
    
    // we want to split the vector into num_threads parts (a part for
    // each thread), then advance each part by log2(num_threads) stages
    // of transformation.
    int stages = 0; // log2() refuses to work on my machine, implemented
    int two_po = 1;
    while(true){
        if(two_po == num_threads){
            break;
            // num_threads is guaranteed to be 2^(stages);
        }
        else{
            stages++;
            two_po *= 2;
        }
    }
    
    // the size of a parallel block:
    int parallel_size = size / num_threads;
    
    // all set, let's do this:
    // copy @vector (for the same reason as in simple_walsh)
    int* new_vector = malloc(sizeof(int) * size);
    if(new_vector == NULL){
      return;
    }

    #pragma omp parallel shared(new_vector, stages)
    {
        // for each thread, define begin and end points
        int thread_id = omp_get_thread_num();
        int point_a = parallel_size * thread_id;
        int point_b = parallel_size * (thread_id + 1); // actual cell is @point_b-1
        
        int i = size;
        while(1 < i && 0 < stages){
            int center = i / 2; // center of this stage's size
            for(int cell = point_a; cell < point_b; cell++){
                if((cell % i) < center){
                    // cell is in first half of a stage size block (i)
                    new_vector[cell] = vector[cell] + vector[cell + center];
                }
                else{
                    // cell in second half
                    new_vector[cell] = vector[cell - center] - vector[cell];
                }
            }
            i = i /2; // next iteration will be on half the size
            #pragma omp barrier
            // all threads finished current iteration
            #pragma omp single
            {
                stages--;
                memcpy(vector, new_vector, sizeof(int) * size);
            }
        }
    }
    free(new_vector);
    // now there are num_threads blocks to work on, uneffected by other threads
    #pragma omp parallel
    {
        // again, define thread jurisdiction
        int thread_id = omp_get_thread_num();
        recursive_fft(vector, thread_id * parallel_size, parallel_size);
    }
}


// modeled after the StackOverflow thread, returns 1 if number of set bits
// of (num & col_id) is even, else returns -1.
int hamming_weight(uint32_t num, uint32_t col_id){
    num &= col_id;
    num = num - ((num >> 1) & 0x55555555);
    num = (num & 0x33333333) + ((num >> 2) & 0x33333333);
    return (((((num + (num >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24) % 2 == 0) ? 1 : -1;
}

/// implemented using one column of the matrix in each thread
void simple_parallel_walsh(int* vector, int size)
{
    // copy @vector, so changes won't affect results for different threads
    int mem_vector[size];
    for(int mi = 0; mi < size; mi++){
        mem_vector[mi] = vector[mi];
    }
    // each thread is responsible for one vector cell
    #pragma omp parallel for
    for(int i = 0 ; i < size ; i++){
        int cell_result = 0; // mutiplication result for vector cell (i)
        for(int j = 0; j < size ; j++){
            cell_result += hamming_weight(j, i) * mem_vector[j];
        }
        vector[i] = cell_result;
    }
}
