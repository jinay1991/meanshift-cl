///
/// @file       meanshift.c
///

// Compilation:
//     - macOS: clang meanshift.c -framework OpenCL
//     - Linux: gcc meanshift.c -lopencl -Lpath/to/opencl
//

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define DATA_SIZE (512)
#define BANDWIDTH (3.0F)

////////////////////////////////////////////////////////////////////////////////

// Simple compute kernel which computes the square of an input array
//
const char *KernelSource_1 =
    "\n"
    "__kernel void algorithm(                                                       \n"
    "   __constant const float2* input_1,                                           \n"
    "   __constant const float2* input_2,                                           \n"
    "   const uint count,                                                           \n"
    "   const float bandwidth,                                                      \n"
    "   __global float2* output)                                                    \n"
    "{                                                                              \n"
    "    int i = get_global_id(0);                                                  \n"
    "    output[i] = input_1[i] + input_2[i];                                       \n"
    "}                                                                              \n"
    "\n";
////////////////////////////////////////////////////////////////////////////////

// Mean Shift Point kernel which computes the mean shift of points
//
const char *KernelSource =
    "\n"
    "__kernel void algorithm(                                                       \n"
    "   __constant const float2* input_1,     // points                             \n"
    "   __constant const float2* input_2,     // original_points                    \n"
    "   const size_t count,                                                         \n"
    "   const float bandwidth,                                                      \n"
    "   __global float2* output)              // shifted_points                     \n"
    "{                                                                              \n"
    "    float pi = 3.14F;                                                          \n"
    "    float base_weight = 1.0F / (bandwidth * sqrt(2.0F * pi));                  \n"
    "    float2 shift = {0.0F, 0.0F};                                               \n"
    "    float scale = 0.0F;                                                        \n"
    "                                                                               \n"
    "    size_t i = get_global_id(0);                                               \n"
    "                                                                               \n"
    "    for (size_t j = 0; j < count; j++)                                         \n"
    "    {                                                                          \n"
    "        float dist = distance(input_1[i], input_2[j]);                         \n"
    "        float weight = base_weight * exp(-0.5F * pow(dist / bandwidth, 2.0F)); \n"
    "                                                                               \n"
    "        shift += input_2[j] * weight;                                          \n"
    "        scale += weight;                                                       \n"
    "    }                                                                          \n"
    "                                                                               \n"
    "    output[i] = shift / scale;                                                 \n"
    "}                                                                              \n"
    "\n";
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    int err;  // error code returned from api calls

    cl_float2 data[DATA_SIZE];     // original data set given to device
    cl_float2 results[DATA_SIZE];  // results returned from device

    unsigned int correct;  // number of correct results returned

    size_t global;  // global domain size for our calculation
    size_t local;   // local domain size for our calculation

    cl_device_id device_id;     // compute device id
    cl_context context;         // compute context
    cl_command_queue commands;  // compute command queue
    cl_program program;         // compute program
    cl_kernel kernel;           // compute kernel
    cl_event event;             // compute profile event

    cl_ulong time_start;  // compute command queue execution time start
    cl_ulong time_end;    // compute command queue execution time end
    double elapsed_time;  // time taken for compute

    cl_mem input_1, input_2;         // device memory used for the input array
    cl_mem output;                   // device memory used for the output array
    cl_float bandwidth = BANDWIDTH;  // device bandwidth

    // Fill our data set with random float values
    //
    int i = 0;
    size_t count = DATA_SIZE;
    for (i = 0; i < count; i++)
    {
        data[i].s[0] = (cl_float)(i);
        data[i].s[1] = (cl_float)(i);

        results[i].s[0] = 0.0F;
        results[i].s[1] = 0.0F;
    }

    printf("Inputs: {\n");
    for (i = 0; i < count; i++)
    {
        printf("%f %f\n", data[i].s[0], data[i].s[1]);
    }
    printf("}\n");

    // Connect to a compute device
    //
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group! %d\n", err);
        return EXIT_FAILURE;
    }

    // Create a compute context
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context || err != CL_SUCCESS)
    {
        printf("Error: Failed to create a compute context! %d\n", err);
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable! %d\n", err);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Create the compute kernel in the program we wish to run
    //
    kernel = clCreateKernel(program, "algorithm", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel! %d\n", err);
        return EXIT_FAILURE;
    }

    // Create the input and output arrays in device memory for our calculation
    //
    input_1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float2) * count, NULL, NULL);
    input_2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float2) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float2) * count, NULL, NULL);
    if (!input_1 || !input_2 || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        return EXIT_FAILURE;
    }

    // Write our data set into the input array in device memory
    //
    err = clEnqueueWriteBuffer(commands, input_1, CL_TRUE, 0, sizeof(cl_float2) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array! %d\n", err);
        return EXIT_FAILURE;
    }
    err = clEnqueueWriteBuffer(commands, input_2, CL_TRUE, 0, sizeof(cl_float2) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array! %d\n", err);
        return EXIT_FAILURE;
    }

    // Set the arguments to our compute kernel
    //
    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_1);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_2);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &count);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_float), &bandwidth);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &output);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return EXIT_FAILURE;
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        return EXIT_FAILURE;
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    // local = ;
    global = count;
    printf("Choosen dim: {global=%ld, local=%ld}\n", global, local);
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, &event);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to execute kernel! %d\n", err);
        return EXIT_FAILURE;
    }

    // Wait for the event commands to get serviced before reading back results
    //
    clWaitForEvents(1, &event);

    // Wait for the command commands to get serviced before reading back results
    //
    clFinish(commands);

    // Read back the results from the device to verify the output
    //
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(cl_float2) * count, results, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        return EXIT_FAILURE;
    }

    // Obtain profiling details
    //
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    elapsed_time = (time_end - time_start) / 1000000.0;

    // Validate our results
    //
    correct = 0;
    for (i = 0; i < count; i++)
    {
        if (results[i].s[0] != 0.0F && results[i].s[1] != 0.0F)
        {
            correct++;
        }
    }

    printf("Results: {\n");
    for (i = 0; i < count; i++)
    {
        printf("%f %f\n", results[i].s[0], results[i].s[1]);
    }
    printf("}\n");

    // Print a brief summary detailing the results
    //
    printf("Computed '%d/%zu' correct values in [%0.3fms]!\n", correct, count, elapsed_time);

    // Shutdown and cleanup
    //
    clReleaseMemObject(input_1);
    clReleaseMemObject(input_2);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}
