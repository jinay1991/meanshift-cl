////////////////
__kernel void multiply(__constant const float* a, __constant const float* b, __global float* result)
{
    size_t i = get_global_id(0);
    result[i] = a[i] * b[i];
}

/*

GPU
0 = x
1 = y
2 = z

                         k               k              k                 k
                        CU1             CU2            CU3               CU4
                indices  0               1              2                 3
call get_global_id()     0               1              2                 3
        result     (a[0] * b[0])   (a[1] * b[1])  (a[2] * b[2])     (a[3] * b[3])
                         0               1              2                 3
*/

/////////////////
#include <stdio.h>

void multiply(float* a, float* b, float* result, int i) { result[i] = a[i] * b[i]; }

int main(int argc, char** argv)
{
    float a[4] = {1, 2, 3, 4};
    float b[4] = {2, 2, 2, 2};

    float result[4] = {0};

    for (int i = 0; i < 4; i++)
    {
        multiply(a, b, result, i);
    }

    multiply(a, b, result, 0);  ///---> thread 1
    multiply(a, b, result, 1);  ///---> thread 2
    multiply(a, b, result, 2);  ///---> thread 3
    multiply(a, b, result, 3);  ///---> thread 4

    // --> Join threads

    // result = { 2, 4, 6, 8 }
    return 0;
}