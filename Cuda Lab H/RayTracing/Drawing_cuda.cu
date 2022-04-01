/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_math.h>

 // includes, cuda
#include <helper_cuda.h>
#include <iostream>
#include "hitable.h"
#include "hitable_list.h"
#include "sphere.h"
#include "vec3.h"
#include "ray.h"

typedef unsigned int uint;
typedef unsigned char uchar;


cudaArray* d_imageArray = 0;
__device__ static int ticks = 1;
__device__ static int colour_index = 1;
__device__ static vec3 sphere_centres[3];
__device__ static vec3 sphere_velocitys[3];
__device__ static vec3 sphere_colours[3];

__device__ vec3 castRay(const ray& r, hitable** world) {
    hit_record rec;

    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        vec3 result = 0.5f * vec3(rec.normal.x() + 1.0f , rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
        return result * (rec.colour/255);
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    }
}
__global__ void create_world(hitable** d_list, hitable** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) 
    {
        //define wall vec3s
        vec3 lef = vec3(-10002.0,        0,        0);
        vec3 rig = vec3( 10002.0,        0,        0);
        vec3 bot = vec3(       0, -10002.0,        0);
        vec3 top = vec3(       0,  10002.0,        0);
        vec3 bac = vec3(       0,        0, -10001.0);

        //define the colours
        vec3 blank = vec3(255, 255, 255);
        int number_of_colours = 6;
        vec3 colour_list[6];
        colour_list[0] = vec3(066, 245, 242);
        colour_list[1] = vec3(245, 066, 224);
        colour_list[2] = vec3(245, 242, 066);
        colour_list[3] = vec3(000, 255, 000);
        colour_list[4] = vec3(000, 000, 255);
        colour_list[5] = vec3(255, 000, 000);

        //initialise sphers statics
        if (sphere_velocitys[0].x() == 0 && sphere_velocitys[0].y() == 0 && sphere_velocitys[0].z() == 0) { sphere_velocitys[0] = vec3(0.01, 0.05, 0); sphere_colours[0] = vec3(150, 150, 150); }
        if (sphere_velocitys[1].x() == 0 && sphere_velocitys[1].y() == 0 && sphere_velocitys[1].z() == 0) { sphere_velocitys[1] = vec3(-0.03, -0.01, 0); sphere_colours[1] = vec3(150, 150, 150); }
        if (sphere_velocitys[2].x() == 0 && sphere_velocitys[2].y() == 0 && sphere_velocitys[2].z() == 0) { sphere_velocitys[2] = vec3(0.05, -0.02, 0); sphere_colours[0] = vec3(150, 150, 150);
        }

        int wall_size = 10000;
        //Create the walls
        *(d_list + 0) = new sphere(lef, blank, vec3(0, 0, 0), wall_size);
        *(d_list + 1) = new sphere(rig, blank, vec3(0, 0, 0), wall_size);
        *(d_list + 2) = new sphere(bot, blank, vec3(0, 0, 0), wall_size);
        *(d_list + 3) = new sphere(top, blank, vec3(0, 0, 0), wall_size);
        *(d_list + 4) = new sphere(bac, blank, vec3(0, 0, 0), wall_size);
        //Modify balls
        for (int i = 0; i < 3; i++)
        {
            //move balls
            sphere_centres[i] += sphere_velocitys[i];
            //Test Collisions
            if (sphere_centres[i].x() - 0.2 <= lef.x() + wall_size || sphere_centres[i].x() + 0.2 >= rig.x() - wall_size)
            {
                sphere_colours[i] = colour_list[colour_index];
                colour_index++;
                if (colour_index > 5)
                {
                    colour_index = 0;
                }
                sphere_velocitys[i] = vec3(sphere_velocitys[i].x() * -1, sphere_velocitys[i].y(), sphere_velocitys[i].z());
            }
            if (sphere_centres[i].y() - 0.2 <= bot.y() + wall_size || sphere_centres[i].y() + 0.2 >= top.y() - wall_size)
            {
                sphere_colours[i] = colour_list[colour_index];
                colour_index++;
                if (colour_index > 5)
                {
                    colour_index = 0;
                }
                sphere_velocitys[i] = vec3(sphere_velocitys[i].x(), sphere_velocitys[i].y() * -1, sphere_velocitys[i].z());
            }
            *(d_list + i + 5) = new sphere(sphere_centres[i], sphere_colours[i], sphere_velocitys[i], 0.2);
        }
        *d_world = new hitable_list(d_list, 8);
    }
}
__global__ void free_world(hitable** d_list, hitable** d_world) {
    delete* (d_list + 0);
    delete* (d_list + 1);
    delete* (d_list + 2);
    delete* (d_list + 3);
    delete* (d_list + 4);
    delete* (d_list + 5);
    delete* d_world;
}

__global__ void d_render(uchar4* d_output, uint width, uint height, hitable** d_world)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint i = y * width + x;

    float u = x / (float)width;
    float v = y / (float)height;

    u = 2.0 * u - 1.0;
    v = -(2.0 * v - 1.0);
    //scale u by aspect ratio
    u *= width / (float)height;

    u *= 2.0;
    v *= 2.0;

    vec3 eye = vec3(0, 0.5, 1.5);
    float distFromEyeToImg = 1.0;
    if ((x < width) && (y < height))
    {
        vec3 pixelPos = vec3(u, v, eye.z() - distFromEyeToImg);
        ray r;
        r.O = eye;
        r.Dir = pixelPos - eye;

        vec3 col = castRay(r, d_world);
        float red = col.x();
        float green = col.y();
        float blue = col.z();
        d_output[i] = make_uchar4(red * 255, green * 255, blue * 255, 0);
    }
}

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

extern "C" void freeTexture() {

    checkCudaErrors(cudaFreeArray(d_imageArray));
}

// render image using CUDA
extern "C" 
    void render(int width, int height,  dim3 blockSize, dim3 gridSize, uchar4 * output) 
{
    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 8 * sizeof(hitable*)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
    create_world << <1, 1 >> > (d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    d_render << <gridSize, blockSize >> > (output, width, height, d_world);
    getLastCudaError("kernel failed");
}
#endif