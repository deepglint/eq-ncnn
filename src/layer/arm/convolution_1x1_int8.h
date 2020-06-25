// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static void conv1x1s1_sgemm_transform_kernel_int8_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    const signed char* kernel = _kernel;

#if __ARM_NEON && __aarch64__
    kernel_tm.create(4*8, inch/4 + inch%4, outch/8 + (outch%8)/4 + outch%4, (size_t)1u);
#else
    kernel_tm.create(4*4, inch/4 + inch%4, outch/4 + outch%4, (size_t)1u);
#endif // __ARM_NEON && __aarch64__    

    int p = 0;
#if __ARM_NEON && __aarch64__
    for (; p+7<outch; p+=8)
    {
        const signed char* kernel0 = kernel + (p+0)*inch;
        const signed char* kernel1 = kernel + (p+1)*inch;
        const signed char* kernel2 = kernel + (p+2)*inch;
        const signed char* kernel3 = kernel + (p+3)*inch;
        const signed char* kernel4 = kernel + (p+4)*inch;
        const signed char* kernel5 = kernel + (p+5)*inch;
        const signed char* kernel6 = kernel + (p+6)*inch;
        const signed char* kernel7 = kernel + (p+7)*inch;

        signed char* ktmp = kernel_tm.channel(p/8);

        for (int q=0; q<inch; q++)
        {
            // kernel0...7 0
            ktmp[0] = kernel0[0];
            ktmp[1] = kernel1[0];
            ktmp[2] = kernel2[0];
            ktmp[3] = kernel3[0];
            ktmp[4] = kernel4[0];
            ktmp[5] = kernel5[0];
            ktmp[6] = kernel6[0];
            ktmp[7] = kernel7[0];

            ktmp += 8;
            kernel0 += 1;
            kernel1 += 1;
            kernel2 += 1;
            kernel3 += 1;
            kernel4 += 1;
            kernel5 += 1;
            kernel6 += 1;
            kernel7 += 1;
        }
    }
#endif // __ARM_NEON && __aarch64__    
    for (; p+3<outch; p+=4)
    {
        const signed char* kernel0 = kernel + (p+0)*inch;
        const signed char* kernel1 = kernel + (p+1)*inch;
        const signed char* kernel2 = kernel + (p+2)*inch;
        const signed char* kernel3 = kernel + (p+3)*inch;

#if __ARM_NEON && __aarch64__
        signed char* ktmp = kernel_tm.channel(p/8 + (p%8)/4);
#else
        signed char* ktmp = kernel_tm.channel(p/4);
#endif // __ARM_NEON && __aarch64__

        for (int q=0; q<inch; q++)
        {
            // kernel0...3 0
            ktmp[0] = kernel0[0];
            ktmp[1] = kernel1[0];
            ktmp[2] = kernel2[0];
            ktmp[3] = kernel3[0];

            ktmp += 4;
            kernel0 += 1;
            kernel1 += 1;
            kernel2 += 1;
            kernel3 += 1;
        }
    }

    for (; p<outch; p++)
    {
        const signed char* kernel0 = kernel + p*inch;

#if __ARM_NEON && __aarch64__
        signed char* ktmp = kernel_tm.channel(p/8 + (p%8)/4 + p%4);
#else
        signed char* ktmp = kernel_tm.channel(p/4 + p%4);
#endif // __ARM_NEON && __aarch64__

        for (int q=0; q<inch; q++)
        {
            ktmp[0] = kernel0[0];
            ktmp++;
            kernel0++;
        }
    }  
}

/*
 * Convolution 1x1 quantized with sgemm int8
 */
static void conv1x1s1_sgemm_int8_e2e_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    int outch = top_blob.c;

    const int size = w * h;

    // interleave
    Mat tmp(8*4, inch/4+inch%4, size/8 + size%8, 1u, opt.workspace_allocator);
    {
        int nn_size = size >> 3;
        int remain_size_start = nn_size << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = ii * 8;

            const signed char* img0 = bottom_blob.channel(0);
            img0 += i;

            signed char* tmpptr = tmp.channel(i/8);

            for (int q=0; q<inch; q++)
            {
#if __ARM_NEON                
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%0, #128]    \n"
                    "ld1    {v0.8b}, [%0]            \n"
                    "st1    {v0.8b}, [%1], #8        \n"
                    : "=r"(img0),   // %0
                      "=r"(tmpptr)  // %1
                    : "0"(img0),
                      "1"(tmpptr)
                    : "cc", "memory", "v0"
                );
#else
                asm volatile(
                    "pld        [%0, #64]     \n"
                    "vld1.s8   {d0}, [%0]     \n"
                    "vst1.s8   {d0}, [%1]!    \n"
                    : "=r"(img0),   // %0
                      "=r"(tmpptr)  // %1
                    : "0"(img0),
                      "1"(tmpptr)
                    : "memory", "d0"
                );
#endif // __aarch64__            
                img0 += bottom_blob.cstep;
#else                
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img0[2];
                tmpptr[3] = img0[3];
                tmpptr[4] = img0[4];
                tmpptr[5] = img0[5];
                tmpptr[6] = img0[6];
                tmpptr[7] = img0[7];

                tmpptr += 8;
                img0 += bottom_blob.cstep;
#endif // __ARM_NEON__                
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=remain_size_start; i<size; i++)
        {
            const signed char* img0 = bottom_blob.channel(0);
            img0 += i;

            signed char* tmpptr = tmp.channel(i/8 + (i%8)/4 + i%4);

            for (int q=0; q<inch; q++)
            {
                tmpptr[0] = img0[0];
                tmpptr++;
                img0 += bottom_blob.cstep;
            }
        }
    }

    // sgemm process
    int nn_outch = 0;
    int remain_outch_start = 0;

#if 1 //__ARM_NEON && __aarch64__
    nn_outch = outch >> 3;
    remain_outch_start = nn_outch << 3;

#if __ARM_NEON
    int16x8_t _int1 = vdupq_n_s16(1);
#endif    

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 8;

        short* outptr0 = top_blob.channel(p);
        short* outptr1 = top_blob.channel(p+1);
        short* outptr2 = top_blob.channel(p+2);
        short* outptr3 = top_blob.channel(p+3);
        short* outptr4 = top_blob.channel(p+4);
        short* outptr5 = top_blob.channel(p+5);
        short* outptr6 = top_blob.channel(p+6);
        short* outptr7 = top_blob.channel(p+7);

        int i = 0;

        for (; i+7<size; i+=8)
        {
            const signed char* tmpptr = tmp.channel(i/8);
            const signed char* kptr = kernel.channel(p/8);

#if 1 //__ARM_NEON && __aarch64__
            asm volatile(
                "eor    v20.16b, v20.16b, v20.16b    \n" // sum0
                "eor    v21.16b, v21.16b, v21.16b    \n" // sum1
                "eor    v22.16b, v22.16b, v22.16b    \n" // sum2
                "eor    v23.16b, v23.16b, v23.16b    \n" // sum3
                "eor    v24.16b, v24.16b, v24.16b    \n" // sum4
                "eor    v25.16b, v25.16b, v25.16b    \n" // sum5
                "eor    v26.16b, v26.16b, v26.16b    \n" // sum6
                "eor    v27.16b, v27.16b, v27.16b    \n" // sum7

                // inch loop
                "lsr    w4, %w20, #2                 \n"// w4 = nn = inch >> 2
                "cmp    w4, #0                       \n"
                "beq    1f                           \n"

                "0:                                  \n"
                "ld1    {v0.16b, v1.16b}, [%9], #32  \n"
                "ld1    {v2.16b, v3.16b}, [%8], #32  \n"

                "dup      v4.8b, v0.b[0]            \n"// k00 - k70
                "dup      v5.8b, v0.b[1]            \n"
                "dup      v6.8b, v0.b[2]            \n"
                "dup      v7.8b, v0.b[3]            \n"
                "dup      v8.8b, v0.b[4]            \n"
                "dup      v9.8b, v0.b[5]            \n"
                "dup      v10.8b, v0.b[6]           \n"
                "dup      v11.8b, v0.b[7]           \n"
                // k0
                "smull    v12.8h, v2.8b, v4.8b      \n"// sum0 += (a00-a70) * k00
                "smull    v13.8h, v2.8b, v5.8b      \n"// sum1 += (a00-a70) * k10
                "smull    v14.8h, v2.8b, v6.8b      \n"// sum2 += (a00-a70) * k20
                "smull    v15.8h, v2.8b, v7.8b      \n"// sum3 += (a00-a70) * k30
                "smull    v16.8h, v2.8b, v8.8b      \n"// sum4 += (a00-a70) * k40
                "smull    v17.8h, v2.8b, v9.8b      \n"// sum5 += (a00-a70) * k50
                "smull    v18.8h, v2.8b, v10.8b     \n"// sum6 += (a00-a70) * k60
                "smull    v19.8h, v2.8b, v11.8b     \n"// sum7 += (a00-a70) * k70

                "dup      v4.16b, v0.b[8]            \n"// k01 - k71
                "dup      v5.16b, v0.b[9]            \n"
                "dup      v6.16b, v0.b[10]           \n"
                "dup      v7.16b, v0.b[11]           \n"
                "dup      v8.16b, v0.b[12]           \n"
                "dup      v9.16b, v0.b[13]           \n"
                "dup      v10.16b, v0.b[14]          \n"
                "dup      v11.16b, v0.b[15]          \n"
                // k1
                "smlal2   v12.8h, v2.16b, v4.16b     \n"// sum0 += (a01-a71) * k01
                "smlal2   v13.8h, v2.16b, v5.16b     \n"// sum1 += (a01-a71) * k11
                "smlal2   v14.8h, v2.16b, v6.16b     \n"// sum2 += (a01-a71) * k21
                "smlal2   v15.8h, v2.16b, v7.16b     \n"// sum3 += (a01-a71) * k31
                "smlal2   v16.8h, v2.16b, v8.16b     \n"// sum4 += (a01-a71) * k41
                "smlal2   v17.8h, v2.16b, v9.16b     \n"// sum5 += (a01-a71) * k51
                "smlal2   v18.8h, v2.16b, v10.16b    \n"// sum6 += (a01-a71) * k61
                "smlal2   v19.8h, v2.16b, v11.16b    \n"// sum7 += (a01-a71) * k71

                "dup      v4.8b, v1.b[0]            \n"// k02 - k72
                "dup      v5.8b, v1.b[1]            \n"
                "dup      v6.8b, v1.b[2]            \n"
                "dup      v7.8b, v1.b[3]            \n"
                "dup      v8.8b, v1.b[4]            \n"
                "dup      v9.8b, v1.b[5]            \n"
                "dup      v10.8b, v1.b[6]           \n"
                "dup      v11.8b, v1.b[7]           \n"
                // k2
                "smlal    v12.8h, v3.8b, v4.8b    \n"// sum0 += (a02-a72) * k02
                "smlal    v13.8h, v3.8b, v5.8b    \n"// sum1 += (a02-a72) * k12
                "smlal    v14.8h, v3.8b, v6.8b    \n"// sum2 += (a02-a72) * k22
                "smlal    v15.8h, v3.8b, v7.8b    \n"// sum3 += (a02-a72) * k32
                "smlal    v16.8h, v3.8b, v8.8b    \n"// sum4 += (a02-a72) * k42
                "smlal    v17.8h, v3.8b, v9.8b    \n"// sum5 += (a02-a72) * k52
                "smlal    v18.8h, v3.8b, v10.8b   \n"// sum6 += (a02-a72) * k62
                "smlal    v19.8h, v3.8b, v11.8b   \n"// sum7 += (a02-a72) * k72

                "subs   w4, w4, #1                   \n"

                "dup      v4.16b, v1.b[8]            \n"// k03 - k73
                "dup      v5.16b, v1.b[9]            \n"
                "dup      v6.16b, v1.b[10]           \n"
                "dup      v7.16b, v1.b[11]           \n"
                "dup      v8.16b, v1.b[12]           \n"
                "dup      v9.16b, v1.b[13]           \n"
                "dup      v10.16b, v1.b[14]          \n"
                "dup      v11.16b, v1.b[15]          \n"
                // k3
                "smlal2    v12.8h, v3.16b, v4.16b    \n"// sum0 += (a03-a73) * k03
                "smlal2    v13.8h, v3.16b, v5.16b    \n"// sum1 += (a03-a73) * k13
                "smlal2    v14.8h, v3.16b, v6.16b    \n"// sum2 += (a03-a73) * k23
                "smlal2    v15.8h, v3.16b, v7.16b    \n"// sum3 += (a03-a73) * k33
                "smlal2    v16.8h, v3.16b, v8.16b    \n"// sum4 += (a03-a73) * k43
                "smlal2    v17.8h, v3.16b, v9.16b    \n"// sum5 += (a03-a73) * k53
                "smlal2    v18.8h, v3.16b, v10.16b   \n"// sum6 += (a03-a73) * k63
                "smlal2    v19.8h, v3.16b, v11.16b   \n"// sum7 += (a03-a73) * k73

                // "add       v12.8h, v12.8h, %21.8h    \n"
                // "add       v13.8h, v13.8h, %21.8h    \n"
                // "add       v14.8h, v14.8h, %21.8h    \n"
                // "add       v15.8h, v15.8h, %21.8h    \n"
                // "add       v16.8h, v16.8h, %21.8h    \n"
                // "add       v17.8h, v17.8h, %21.8h    \n"
                // "add       v18.8h, v18.8h, %21.8h    \n"
                // "add       v19.8h, v19.8h, %21.8h    \n"

                // "sshr      v12.8h, v12.8h, #1        \n"
                // "sshr      v13.8h, v13.8h, #1        \n"
                // "sshr      v14.8h, v14.8h, #1        \n"
                // "sshr      v15.8h, v15.8h, #1        \n"
                // "sshr      v16.8h, v16.8h, #1        \n"
                // "sshr      v17.8h, v17.8h, #1        \n"
                // "sshr      v18.8h, v18.8h, #1        \n"
                // "sshr      v19.8h, v19.8h, #1        \n"
                
                "sqadd     v20.8h, v20.8h, v12.8h    \n"
                "sqadd     v21.8h, v21.8h, v13.8h    \n"
                "sqadd     v22.8h, v22.8h, v14.8h    \n"
                "sqadd     v23.8h, v23.8h, v15.8h    \n"
                "sqadd     v24.8h, v24.8h, v16.8h    \n"
                "sqadd     v25.8h, v25.8h, v17.8h    \n"
                "sqadd     v26.8h, v26.8h, v18.8h    \n"
                "sqadd     v27.8h, v27.8h, v19.8h    \n"

                "bne    0b                           \n"

                "1:                                  \n"

                // remain loop
                "and    w4, %w20, #3                 \n"// w4 = remain = inch & 3;
                "cmp    w4, #0                       \n"
                "beq    3f                           \n"

                "2:                                  \n"
                "ld1    {v0.8b}, [%9], #8            \n"
                "ld1    {v2.8b}, [%8], #8            \n"

                "subs   w4, w4, #1                   \n"

                "dup      v4.8b, v0.b[0]            \n"// k00 - k70
                "dup      v5.8b, v0.b[1]            \n"
                "dup      v6.8b, v0.b[2]            \n"
                "dup      v7.8b, v0.b[3]            \n"
                "dup      v8.8b, v0.b[4]            \n"
                "dup      v9.8b, v0.b[5]            \n"
                "dup      v10.8b, v0.b[6]           \n"
                "dup      v11.8b, v0.b[7]           \n"

                // k0
                "smull    v12.8h, v2.8b, v4.8b      \n"// sum0 += (a00-a70) * k00
                "smull    v13.8h, v2.8b, v5.8b      \n"// sum1 += (a00-a70) * k10
                "smull    v14.8h, v2.8b, v6.8b      \n"// sum2 += (a00-a70) * k20
                "smull    v15.8h, v2.8b, v7.8b      \n"// sum3 += (a00-a70) * k30
                "smull    v16.8h, v2.8b, v8.8b      \n"// sum4 += (a00-a70) * k40
                "smull    v17.8h, v2.8b, v9.8b      \n"// sum5 += (a00-a70) * k50
                "smull    v18.8h, v2.8b, v10.8b     \n"// sum6 += (a00-a70) * k60
                "smull    v19.8h, v2.8b, v11.8b     \n"// sum7 += (a00-a70) * k70

                // "add       v12.8h, v12.8h, %21.8h    \n"
                // "add       v13.8h, v13.8h, %21.8h    \n"
                // "add       v14.8h, v14.8h, %21.8h    \n"
                // "add       v15.8h, v15.8h, %21.8h    \n"
                // "add       v16.8h, v16.8h, %21.8h    \n"
                // "add       v17.8h, v17.8h, %21.8h    \n"
                // "add       v18.8h, v18.8h, %21.8h    \n"
                // "add       v19.8h, v19.8h, %21.8h    \n"

                // "sshr      v12.8h, v12.8h, #1        \n"
                // "sshr      v13.8h, v13.8h, #1        \n"
                // "sshr      v14.8h, v14.8h, #1        \n"
                // "sshr      v15.8h, v15.8h, #1        \n"
                // "sshr      v16.8h, v16.8h, #1        \n"
                // "sshr      v17.8h, v17.8h, #1        \n"
                // "sshr      v18.8h, v18.8h, #1        \n"
                // "sshr      v19.8h, v19.8h, #1        \n"

                "sqadd     v20.8h, v20.8h, v12.8h    \n"
                "sqadd     v21.8h, v21.8h, v13.8h    \n"
                "sqadd     v22.8h, v22.8h, v14.8h    \n"
                "sqadd     v23.8h, v23.8h, v15.8h    \n"
                "sqadd     v24.8h, v24.8h, v16.8h    \n"
                "sqadd     v25.8h, v25.8h, v17.8h    \n"
                "sqadd     v26.8h, v26.8h, v18.8h    \n"
                "sqadd     v27.8h, v27.8h, v19.8h    \n"

                "bne    2b                           \n"

                "3:                                  \n"

                "st1    {v20.8h}, [%0], #16  \n"
                "st1    {v21.8h}, [%1], #16  \n"
                "st1    {v22.8h}, [%2], #16  \n"
                "st1    {v23.8h}, [%3], #16  \n"
                "st1    {v24.8h}, [%4], #16  \n"
                "st1    {v25.8h}, [%5], #16  \n"
                "st1    {v26.8h}, [%6], #16  \n"
                "st1    {v27.8h}, [%7], #16  \n"

                : "=r"(outptr0),    // %0
                  "=r"(outptr1),    // %1
                  "=r"(outptr2),    // %2
                  "=r"(outptr3),    // %3
                  "=r"(outptr4),    // %4
                  "=r"(outptr5),    // %5
                  "=r"(outptr6),    // %6
                  "=r"(outptr7),    // %7
                  "=r"(tmpptr),     // %8
                  "=r"(kptr)        // %9
                : "0"(outptr0),
                  "1"(outptr1),
                  "2"(outptr2),
                  "3"(outptr3),
                  "4"(outptr4),
                  "5"(outptr5),
                  "6"(outptr6),
                  "7"(outptr7),
                  "8"(tmpptr),
                  "9"(kptr),
                  "r"(inch),        // %20
                  "w"(_int1)        // %21
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27"
            );
#else
            short sum0_0 = 0;
            short sum0_1 = 0;
            short sum0_2 = 0;
            short sum0_3 = 0;
            short sum0_4 = 0;
            short sum0_5 = 0;
            short sum0_6 = 0;
            short sum0_7 = 0;

            short sum1_0 = 0;
            short sum1_1 = 0;
            short sum1_2 = 0;
            short sum1_3 = 0;
            short sum1_4 = 0;
            short sum1_5 = 0;
            short sum1_6 = 0;
            short sum1_7 = 0;

            short sum2_0 = 0;
            short sum2_1 = 0;
            short sum2_2 = 0;
            short sum2_3 = 0;
            short sum2_4 = 0;
            short sum2_5 = 0;
            short sum2_6 = 0;
            short sum2_7 = 0;

            short sum3_0 = 0;
            short sum3_1 = 0;
            short sum3_2 = 0;
            short sum3_3 = 0;
            short sum3_4 = 0;
            short sum3_5 = 0;
            short sum3_6 = 0;
            short sum3_7 = 0;

            short sum4_0 = 0;
            short sum4_1 = 0;
            short sum4_2 = 0;
            short sum4_3 = 0;
            short sum4_4 = 0;
            short sum4_5 = 0;
            short sum4_6 = 0;
            short sum4_7 = 0;

            short sum5_0 = 0;
            short sum5_1 = 0;
            short sum5_2 = 0;
            short sum5_3 = 0;
            short sum5_4 = 0;
            short sum5_5 = 0;
            short sum5_6 = 0;
            short sum5_7 = 0;

            short sum6_0 = 0;
            short sum6_1 = 0;
            short sum6_2 = 0;
            short sum6_3 = 0;
            short sum6_4 = 0;
            short sum6_5 = 0;
            short sum6_6 = 0;
            short sum6_7 = 0;

            short sum7_0 = 0;
            short sum7_1 = 0;
            short sum7_2 = 0;
            short sum7_3 = 0;
            short sum7_4 = 0;
            short sum7_5 = 0;
            short sum7_6 = 0;
            short sum7_7 = 0;

            for (int q=0; q<inch; q++)
            {
				sum0_0 = saturate2int16((int)sum0_0 + ((((short)tmpptr[0] * kptr[0]) + 1) >> 1));
                sum0_1 = saturate2int16((int)sum0_1 + ((((short)tmpptr[1] * kptr[0]) + 1) >> 1));
                sum0_2 = saturate2int16((int)sum0_2 + ((((short)tmpptr[2] * kptr[0]) + 1) >> 1));
                sum0_3 = saturate2int16((int)sum0_3 + ((((short)tmpptr[3] * kptr[0]) + 1) >> 1));
                sum0_4 = saturate2int16((int)sum0_4 + ((((short)tmpptr[4] * kptr[0]) + 1) >> 1));
                sum0_5 = saturate2int16((int)sum0_5 + ((((short)tmpptr[5] * kptr[0]) + 1) >> 1));
                sum0_6 = saturate2int16((int)sum0_6 + ((((short)tmpptr[6] * kptr[0]) + 1) >> 1));
                sum0_7 = saturate2int16((int)sum0_7 + ((((short)tmpptr[7] * kptr[0]) + 1) >> 1));

                sum1_0 = saturate2int16((int)sum1_0 + ((((short)tmpptr[0] * kptr[1]) + 1) >> 1));
                sum1_1 = saturate2int16((int)sum1_1 + ((((short)tmpptr[1] * kptr[1]) + 1) >> 1));
                sum1_2 = saturate2int16((int)sum1_2 + ((((short)tmpptr[2] * kptr[1]) + 1) >> 1));
                sum1_3 = saturate2int16((int)sum1_3 + ((((short)tmpptr[3] * kptr[1]) + 1) >> 1));
                sum1_4 = saturate2int16((int)sum1_4 + ((((short)tmpptr[4] * kptr[1]) + 1) >> 1));
                sum1_5 = saturate2int16((int)sum1_5 + ((((short)tmpptr[5] * kptr[1]) + 1) >> 1));
                sum1_6 = saturate2int16((int)sum1_6 + ((((short)tmpptr[6] * kptr[1]) + 1) >> 1));
                sum1_7 = saturate2int16((int)sum1_7 + ((((short)tmpptr[7] * kptr[1]) + 1) >> 1));

                sum2_0 = saturate2int16((int)sum2_0 + ((((short)tmpptr[0] * kptr[2]) + 1) >> 1));
                sum2_1 = saturate2int16((int)sum2_1 + ((((short)tmpptr[1] * kptr[2]) + 1) >> 1));
                sum2_2 = saturate2int16((int)sum2_2 + ((((short)tmpptr[2] * kptr[2]) + 1) >> 1));
                sum2_3 = saturate2int16((int)sum2_3 + ((((short)tmpptr[3] * kptr[2]) + 1) >> 1));
                sum2_4 = saturate2int16((int)sum2_4 + ((((short)tmpptr[4] * kptr[2]) + 1) >> 1));
                sum2_5 = saturate2int16((int)sum2_5 + ((((short)tmpptr[5] * kptr[2]) + 1) >> 1));
                sum2_6 = saturate2int16((int)sum2_6 + ((((short)tmpptr[6] * kptr[2]) + 1) >> 1));
                sum2_7 = saturate2int16((int)sum2_7 + ((((short)tmpptr[7] * kptr[2]) + 1) >> 1));

                sum3_0 = saturate2int16((int)sum3_0 + ((((short)tmpptr[0] * kptr[3]) + 1) >> 1));
                sum3_1 = saturate2int16((int)sum3_1 + ((((short)tmpptr[1] * kptr[3]) + 1) >> 1));
                sum3_2 = saturate2int16((int)sum3_2 + ((((short)tmpptr[2] * kptr[3]) + 1) >> 1));
                sum3_3 = saturate2int16((int)sum3_3 + ((((short)tmpptr[3] * kptr[3]) + 1) >> 1));
                sum3_4 = saturate2int16((int)sum3_4 + ((((short)tmpptr[4] * kptr[3]) + 1) >> 1));
                sum3_5 = saturate2int16((int)sum3_5 + ((((short)tmpptr[5] * kptr[3]) + 1) >> 1));
                sum3_6 = saturate2int16((int)sum3_6 + ((((short)tmpptr[6] * kptr[3]) + 1) >> 1));
                sum3_7 = saturate2int16((int)sum3_7 + ((((short)tmpptr[7] * kptr[3]) + 1) >> 1));

                sum4_0 = saturate2int16((int)sum4_0 + ((((short)tmpptr[0] * kptr[4]) + 1) >> 1));
                sum4_1 = saturate2int16((int)sum4_1 + ((((short)tmpptr[1] * kptr[4]) + 1) >> 1));
                sum4_2 = saturate2int16((int)sum4_2 + ((((short)tmpptr[2] * kptr[4]) + 1) >> 1));
                sum4_3 = saturate2int16((int)sum4_3 + ((((short)tmpptr[3] * kptr[4]) + 1) >> 1));
                sum4_4 = saturate2int16((int)sum4_4 + ((((short)tmpptr[4] * kptr[4]) + 1) >> 1));
                sum4_5 = saturate2int16((int)sum4_5 + ((((short)tmpptr[5] * kptr[4]) + 1) >> 1));
                sum4_6 = saturate2int16((int)sum4_6 + ((((short)tmpptr[6] * kptr[4]) + 1) >> 1));
                sum4_7 = saturate2int16((int)sum4_7 + ((((short)tmpptr[7] * kptr[4]) + 1) >> 1));

                sum5_0 = saturate2int16((int)sum5_0 + ((((short)tmpptr[0] * kptr[5]) + 1) >> 1));
                sum5_1 = saturate2int16((int)sum5_1 + ((((short)tmpptr[1] * kptr[5]) + 1) >> 1));
                sum5_2 = saturate2int16((int)sum5_2 + ((((short)tmpptr[2] * kptr[5]) + 1) >> 1));
                sum5_3 = saturate2int16((int)sum5_3 + ((((short)tmpptr[3] * kptr[5]) + 1) >> 1));
                sum5_4 = saturate2int16((int)sum5_4 + ((((short)tmpptr[4] * kptr[5]) + 1) >> 1));
                sum5_5 = saturate2int16((int)sum5_5 + ((((short)tmpptr[5] * kptr[5]) + 1) >> 1));
                sum5_6 = saturate2int16((int)sum5_6 + ((((short)tmpptr[6] * kptr[5]) + 1) >> 1));
                sum5_7 = saturate2int16((int)sum5_7 + ((((short)tmpptr[7] * kptr[5]) + 1) >> 1));

                sum6_0 = saturate2int16((int)sum6_0 + ((((short)tmpptr[0] * kptr[6]) + 1) >> 1));
                sum6_1 = saturate2int16((int)sum6_1 + ((((short)tmpptr[1] * kptr[6]) + 1) >> 1));
                sum6_2 = saturate2int16((int)sum6_2 + ((((short)tmpptr[2] * kptr[6]) + 1) >> 1));
                sum6_3 = saturate2int16((int)sum6_3 + ((((short)tmpptr[3] * kptr[6]) + 1) >> 1));
                sum6_4 = saturate2int16((int)sum6_4 + ((((short)tmpptr[4] * kptr[6]) + 1) >> 1));
                sum6_5 = saturate2int16((int)sum6_5 + ((((short)tmpptr[5] * kptr[6]) + 1) >> 1));
                sum6_6 = saturate2int16((int)sum6_6 + ((((short)tmpptr[6] * kptr[6]) + 1) >> 1));
                sum6_7 = saturate2int16((int)sum6_7 + ((((short)tmpptr[7] * kptr[6]) + 1) >> 1));

                sum7_0 = saturate2int16((int)sum7_0 + ((((short)tmpptr[0] * kptr[7]) + 1) >> 1));
                sum7_1 = saturate2int16((int)sum7_1 + ((((short)tmpptr[1] * kptr[7]) + 1) >> 1));
                sum7_2 = saturate2int16((int)sum7_2 + ((((short)tmpptr[2] * kptr[7]) + 1) >> 1));
                sum7_3 = saturate2int16((int)sum7_3 + ((((short)tmpptr[3] * kptr[7]) + 1) >> 1));
                sum7_4 = saturate2int16((int)sum7_4 + ((((short)tmpptr[4] * kptr[7]) + 1) >> 1));
                sum7_5 = saturate2int16((int)sum7_5 + ((((short)tmpptr[5] * kptr[7]) + 1) >> 1));
                sum7_6 = saturate2int16((int)sum7_6 + ((((short)tmpptr[6] * kptr[7]) + 1) >> 1));
                sum7_7 = saturate2int16((int)sum7_7 + ((((short)tmpptr[7] * kptr[7]) + 1) >> 1));

                tmpptr += 8;
                kptr += 8;
            }

            outptr0[0] = sum0_0;
            outptr0[1] = sum0_1;
            outptr0[2] = sum0_2;
            outptr0[3] = sum0_3;
            outptr0[4] = sum0_4;
            outptr0[5] = sum0_5;
            outptr0[6] = sum0_6;
            outptr0[7] = sum0_7;

            outptr1[0] = sum1_0;
            outptr1[1] = sum1_1;
            outptr1[2] = sum1_2;
            outptr1[3] = sum1_3;
            outptr1[4] = sum1_4;
            outptr1[5] = sum1_5;
            outptr1[6] = sum1_6;
            outptr1[7] = sum1_7;

            outptr2[0] = sum2_0;
            outptr2[1] = sum2_1;
            outptr2[2] = sum2_2;
            outptr2[3] = sum2_3;
            outptr2[4] = sum2_4;
            outptr2[5] = sum2_5;
            outptr2[6] = sum2_6;
            outptr2[7] = sum2_7;

            outptr3[0] = sum3_0;
            outptr3[1] = sum3_1;
            outptr3[2] = sum3_2;
            outptr3[3] = sum3_3;
            outptr3[4] = sum3_4;
            outptr3[5] = sum3_5;
            outptr3[6] = sum3_6;
            outptr3[7] = sum3_7;

            outptr4[0] = sum4_0;
            outptr4[1] = sum4_1;
            outptr4[2] = sum4_2;
            outptr4[3] = sum4_3;
            outptr4[4] = sum4_4;
            outptr4[5] = sum4_5;
            outptr4[6] = sum4_6;
            outptr4[7] = sum4_7;

            outptr5[0] = sum5_0;
            outptr5[1] = sum5_1;
            outptr5[2] = sum5_2;
            outptr5[3] = sum5_3;
            outptr5[4] = sum5_4;
            outptr5[5] = sum5_5;
            outptr5[6] = sum5_6;
            outptr5[7] = sum5_7;

            outptr6[0] = sum6_0;
            outptr6[1] = sum6_1;
            outptr6[2] = sum6_2;
            outptr6[3] = sum6_3;
            outptr6[4] = sum6_4;
            outptr6[5] = sum6_5;
            outptr6[6] = sum6_6;
            outptr6[7] = sum6_7;

            outptr7[0] = sum7_0;
            outptr7[1] = sum7_1;
            outptr7[2] = sum7_2;
            outptr7[3] = sum7_3;
            outptr7[4] = sum7_4;
            outptr7[5] = sum7_5;
            outptr7[6] = sum7_6;
            outptr7[7] = sum7_7;

            outptr0 += 8;
            outptr1 += 8;
            outptr2 += 8;
            outptr3 += 8;
            outptr4 += 8;
            outptr5 += 8;
            outptr6 += 8;
            outptr7 += 8;
#endif            
        }

        for (; i<size; i++)
        {
            const signed char* tmpptr = tmp.channel(i/8 + i%8);
            const signed char* kptr = kernel.channel(p/8);

#if 1 // __ARM_NEON && __aarch64__
            asm volatile(
                "eor    v22.16b, v22.16b, v22.16b    \n" // sum0_7

                // inch loop
                "lsr    w4, %w20, #3                 \n"// w4 = nn = inch >> 3
                "cmp    w4, #0                       \n"
                "beq    1f                           \n"

                "0:                                  \n"

                //"prfm   pldl1keep, [%9, #128]                       \n" // k
                "ld1    {v0.8b, v1.8b, v2.8b, v3.8b}, [%9], #32     \n"
                "ld1    {v4.8b, v5.8b, v6.8b, v7.8b}, [%9], #32     \n"

                //"prfm   pldl1keep, [%8, #64]                        \n" // d
                "ld1    {v8.8b}, [%8]                               \n"
                "add    %8, %8, #8                                  \n"

                "dup      v9.8b, v8.b[0]              \n" // a00
                "dup      v10.8b, v8.b[1]             \n" // a10
                "dup      v11.8b, v8.b[2]             \n" // a20
                "dup      v12.8b, v8.b[3]             \n" // a30
                "dup      v13.8b, v8.b[4]             \n" // a40
                "dup      v14.8b, v8.b[5]             \n" // a50
                "dup      v15.8b, v8.b[6]             \n" // a60
                "dup      v16.8b, v8.b[7]             \n" // a70
                //
                "smull    v17.8h, v0.8b, v9.8b        \n"// sum0 += (k00-k70) * a00
                "smull    v18.8h, v1.8b, v10.8b       \n"// sum1 += (k01-k71) * a10
                "smull    v19.8h, v2.8b, v11.8b       \n"// sum2 += (k02-k72) * a20
                "smull    v20.8h, v3.8b, v12.8b       \n"// sum3 += (k03-k73) * a30
                "smlal    v17.8h, v4.8b, v13.8b       \n"// sum4 += (k04-k74) * a40
                "smlal    v18.8h, v5.8b, v14.8b       \n"// sum5 += (k05-k75) * a50
                "smlal    v19.8h, v6.8b, v15.8b       \n"// sum6 += (k06-k76) * a60
                "smlal    v20.8h, v7.8b, v16.8b       \n"// sum7 += (k07-k77) * a70

                "subs   w4, w4, #1                    \n"

                "add      v17.8h, v17.8h, v18.8h      \n"
                "add      v20.8h, v20.8h, v19.8h      \n"
                "add      v21.8h, v17.8h, v20.8h      \n"

                // "add      v21.8h, v21.8h, %21.8h      \n"
                // "sshr     v21.8h, v21.8h, #1          \n"

                "sqadd    v22.8h, v22.8h, v21.8h      \n"

                "bne    0b                            \n"

                "1:                                   \n"

                // remain loop
                "and    w4, %w20, #7                 \n"// w4 = remain = inch & 7;
                "cmp    w4, #0                       \n"
                "beq    3f                           \n"

                "2:                                  \n"

                //"prfm   pldl1keep, [%9, #128]      \n"
                "ld1    {v0.8b}, [%9], #8            \n"// k
                "ld1    {v8.8b}, [%8]                \n"// d
                "add    %8, %8, #1                   \n"

                "dup    v9.8b, v8.b[0]               \n" // a00

                "subs   w4, w4, #1                   \n"

                // k0
                "smull    v17.8h, v0.8b, v9.8b       \n"// sum0 += (k00-k70) * a00

                // "add      v17.8h, v17.8h, %21.8h     \n"
                // "sshr     v17.8h, v17.8h, #1         \n"

                "sqadd    v22.8h, v22.8h, v17.8h     \n"

                "bne    2b                           \n"

                "3:                                  \n"

                "st1    {v22.h}[0], [%0], #2         \n"
                "st1    {v22.h}[1], [%1], #2         \n"
                "st1    {v22.h}[2], [%2], #2         \n"
                "st1    {v22.h}[3], [%3], #2         \n"
                "st1    {v22.h}[4], [%4], #2         \n"
                "st1    {v22.h}[5], [%5], #2         \n"
                "st1    {v22.h}[6], [%6], #2         \n"
                "st1    {v22.h}[7], [%7], #2         \n"

                : "=r"(outptr0),    // %0
                  "=r"(outptr1),    // %1
                  "=r"(outptr2),    // %2
                  "=r"(outptr3),    // %3
                  "=r"(outptr4),    // %4
                  "=r"(outptr5),    // %5
                  "=r"(outptr6),    // %6
                  "=r"(outptr7),    // %7
                  "=r"(tmpptr),     // %8
                  "=r"(kptr)        // %9
                : "0"(outptr0),
                  "1"(outptr1),
                  "2"(outptr2),
                  "3"(outptr3),
                  "4"(outptr4),
                  "5"(outptr5),
                  "6"(outptr6),
                  "7"(outptr7),
                  "8"(tmpptr),
                  "9"(kptr),
                  "r"(inch),        // %20
                  "w"(_int1)        // %21
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            );
#else
            short sum0 = 0;
            short sum1 = 0;
            short sum2 = 0;
            short sum3 = 0;
            short sum4 = 0;
            short sum5 = 0;
            short sum6 = 0;
            short sum7 = 0;            

            for (int q=0; q<inch; q++)
            {
                sum0 = saturate2int16((int)sum0 + ((((short)tmpptr[0] * kptr[0]) + 1) >> 1));
                sum1 = saturate2int16((int)sum1 + ((((short)tmpptr[0] * kptr[1]) + 1) >> 1));
                sum2 = saturate2int16((int)sum2 + ((((short)tmpptr[0] * kptr[2]) + 1) >> 1));
                sum3 = saturate2int16((int)sum3 + ((((short)tmpptr[0] * kptr[3]) + 1) >> 1));
                sum4 = saturate2int16((int)sum4 + ((((short)tmpptr[0] * kptr[4]) + 1) >> 1));
                sum5 = saturate2int16((int)sum5 + ((((short)tmpptr[0] * kptr[5]) + 1) >> 1));
                sum6 = saturate2int16((int)sum6 + ((((short)tmpptr[0] * kptr[6]) + 1) >> 1));
                sum7 = saturate2int16((int)sum7 + ((((short)tmpptr[0] * kptr[7]) + 1) >> 1));

                tmpptr++;
                kptr += 8;
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr2[0] = sum2;
            outptr3[0] = sum3;
            outptr4[0] = sum4;
            outptr5[0] = sum5;
            outptr6[0] = sum6;
            outptr7[0] = sum7;

            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;
            outptr4++;
            outptr5++;
            outptr6++;
            outptr7++;
#endif            
        }
    }
#endif // __ARM_NEON && __aarch64__ 

    nn_outch = (outch - remain_outch_start) >> 2;  

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = remain_outch_start + pp * 4;

        short* outptr0 = top_blob.channel(p);
        short* outptr1 = top_blob.channel(p+1);
        short* outptr2 = top_blob.channel(p+2);
        short* outptr3 = top_blob.channel(p+3);

        int i = 0;

        for (; i+7<size; i+=8)
        {
            const signed char* tmpptr = tmp.channel(i/8);
#if 1 //__ARM_NEON && __aarch64__
            const signed char* kptr = kernel.channel(p/8 + (p%8)/4);
#else
            const signed char* kptr = kernel.channel(p/4);
#endif // __ARM_NEON && __aarch64__
#if 1 // __ARM_NEON
#if __aarch64__
            asm volatile(
                "eor    v20.16b, v20.16b, v20.16b    \n" // sum0
                "eor    v21.16b, v21.16b, v21.16b    \n" // sum1
                "eor    v22.16b, v22.16b, v22.16b    \n" // sum2
                "eor    v23.16b, v23.16b, v23.16b    \n" // sum3

                // inch loop
                "lsr    w4, %w12, #2                 \n"// w4 = nn = inch >> 2
                "cmp    w4, #0                       \n"
                "beq    1f                           \n"

                "0:                                  \n"
                "ld1    {v0.16b}, [%5], #16          \n"
                "ld1    {v2.16b, v3.16b}, [%4], #32  \n"

                "dup      v4.8b, v0.b[0]            \n"// k00 - k30
                "dup      v5.8b, v0.b[1]            \n"
                "dup      v6.8b, v0.b[2]            \n"
                "dup      v7.8b, v0.b[3]            \n"
                "dup      v8.16b, v0.b[4]           \n"// k01 - k31
                "dup      v9.16b, v0.b[5]           \n"
                "dup      v10.16b, v0.b[6]          \n"
                "dup      v11.16b, v0.b[7]          \n"

                // k0
                "smull    v12.8h, v2.8b, v4.8b      \n"// sum0 += (a00-a70) * k00
                "smull    v13.8h, v2.8b, v5.8b      \n"// sum1 += (a00-a70) * k10
                "smull    v14.8h, v2.8b, v6.8b      \n"// sum2 += (a00-a70) * k20
                "smull    v15.8h, v2.8b, v7.8b      \n"// sum3 += (a00-a70) * k30
                // k1
                "smlal2   v12.8h, v2.16b, v8.16b    \n"// sum0 += (a01-a71) * k01
                "smlal2   v13.8h, v2.16b, v9.16b    \n"// sum1 += (a01-a71) * k11
                "smlal2   v14.8h, v2.16b, v10.16b   \n"// sum2 += (a01-a71) * k21
                "smlal2   v15.8h, v2.16b, v11.16b   \n"// sum3 += (a01-a71) * k31

                "dup      v4.8b, v0.b[8]             \n"// k02 - k32
                "dup      v5.8b, v0.b[9]             \n"
                "dup      v6.8b, v0.b[10]            \n"
                "dup      v7.8b, v0.b[11]            \n"
                "dup      v8.16b, v0.b[12]           \n"// k03 - k33
                "dup      v9.16b, v0.b[13]           \n"
                "dup      v10.16b, v0.b[14]          \n"
                "dup      v11.16b, v0.b[15]          \n"
                // k2
                "smlal    v12.8h, v3.8b, v4.8b    \n"// sum0 += (a02-a72) * k02
                "smlal    v13.8h, v3.8b, v5.8b    \n"// sum1 += (a02-a72) * k12
                "smlal    v14.8h, v3.8b, v6.8b    \n"// sum2 += (a02-a72) * k22
                "smlal    v15.8h, v3.8b, v7.8b    \n"// sum3 += (a02-a72) * k32

                "subs   w4, w4, #1                   \n"
                // k3
                "smlal2    v12.8h, v3.16b, v8.16b    \n"// sum0 += (a03-a73) * k03
                "smlal2    v13.8h, v3.16b, v9.16b    \n"// sum1 += (a03-a73) * k13
                "smlal2    v14.8h, v3.16b, v10.16b   \n"// sum2 += (a03-a73) * k23
                "smlal2    v15.8h, v3.16b, v11.16b   \n"// sum3 += (a03-a73) * k33

                // "add       v12.8h, v12.8h, %13.8h    \n"
                // "add       v13.8h, v13.8h, %13.8h    \n"
                // "add       v14.8h, v14.8h, %13.8h    \n"
                // "add       v15.8h, v15.8h, %13.8h    \n"
                // "sshr      v12.8h, v12.8h, #1        \n"
                // "sshr      v13.8h, v13.8h, #1        \n"
                // "sshr      v14.8h, v14.8h, #1        \n"
                // "sshr      v15.8h, v15.8h, #1        \n"

                "sqadd     v20.8h, v20.8h, v12.8h    \n"
                "sqadd     v21.8h, v21.8h, v13.8h    \n"
                "sqadd     v22.8h, v22.8h, v14.8h    \n"
                "sqadd     v23.8h, v23.8h, v15.8h    \n"                

                "bne    0b                           \n"

                "1:                                  \n"

                // remain loop
                "and    w4, %w12, #3                 \n"// w4 = remain = inch & 3;
                "cmp    w4, #0                       \n"
                "beq    3f                           \n"

                "2:                                  \n"
                "ld1    {v0.8b}, [%5]                \n"
                "ld1    {v2.8b}, [%4], #8            \n"
                "add    %5, %5, #4                   \n"

                "dup      v4.8b, v0.b[0]            \n"// k00 - k70
                "dup      v5.8b, v0.b[1]            \n"
                "dup      v6.8b, v0.b[2]            \n"
                "dup      v7.8b, v0.b[3]            \n"

                "subs   w4, w4, #1                   \n"

                // k0
                "smull    v12.8h, v2.8b, v4.8b      \n"// sum0 += (a00-a70) * k00
                "smull    v13.8h, v2.8b, v5.8b      \n"// sum1 += (a00-a70) * k10
                "smull    v14.8h, v2.8b, v6.8b      \n"// sum2 += (a00-a70) * k20
                "smull    v15.8h, v2.8b, v7.8b      \n"// sum3 += (a00-a70) * k30

                // "add       v12.8h, v12.8h, %13.8h    \n"
                // "add       v13.8h, v13.8h, %13.8h    \n"
                // "add       v14.8h, v14.8h, %13.8h    \n"
                // "add       v15.8h, v15.8h, %13.8h    \n"
                // "sshr      v12.8h, v12.8h, #1        \n"
                // "sshr      v13.8h, v13.8h, #1        \n"
                // "sshr      v14.8h, v14.8h, #1        \n"
                // "sshr      v15.8h, v15.8h, #1        \n"

                "sqadd     v20.8h, v20.8h, v12.8h    \n"
                "sqadd     v21.8h, v21.8h, v13.8h    \n"
                "sqadd     v22.8h, v22.8h, v14.8h    \n"
                "sqadd     v23.8h, v23.8h, v15.8h    \n"                

                "bne    2b                           \n"

                "3:                                  \n"

                "st1    {v20.8h}, [%0], #16          \n"
                "st1    {v21.8h}, [%1], #16          \n"
                "st1    {v22.8h}, [%2], #16          \n"
                "st1    {v23.8h}, [%3], #16          \n"

                : "=r"(outptr0),    // %0
                  "=r"(outptr1),    // %1
                  "=r"(outptr2),    // %2
                  "=r"(outptr3),    // %3
                  "=r"(tmpptr),     // %4
                  "=r"(kptr)        // %5
                : "0"(outptr0),
                  "1"(outptr1),
                  "2"(outptr2),
                  "3"(outptr3),
                  "4"(tmpptr),
                  "5"(kptr),
                  "r"(inch),        // %12
                  "w"(_int1)        // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
            );
#else
            asm volatile(
                // inch loop
                "vmov.s32    q6, #0            \n"
                "vmov.s32    q7, #0            \n"
                "vmov.s32    q8, #0            \n"
                "vmov.s32    q9, #0            \n"
                "vmov.s32    q10, #0           \n"
                "vmov.s32    q11, #0           \n"
                "vmov.s32    q12, #0           \n"
                "vmov.s32    q13, #0           \n"

                "lsr         r4, %12, #2       \n"// r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"
                
                "0:                            \n"// for(; nn != 0; nn--)
                "pld         [%4, #128]        \n"
                "vld1.s8     {d4-d7}, [%4]!    \n"// tmpr a00-a07,a10-a17,a20-a27,a30-a37    a(inch)(data)
                "vmovl.s8    q5, d7            \n"// a30-a37
                "vmovl.s8    q4, d6            \n"// a20-a27
                "vmovl.s8    q3, d5            \n"// a10-a17
                "vmovl.s8    q2, d4            \n"// a00-a07

                "vld1.s8     {d0-d1}, [%5]!    \n"// kptr k00-k30,k01-k31,k02-k32,k03-k33    k(outch)(inch)
                "vmovl.s8    q1, d1            \n"// k02-k32,k03-k33
                "vmovl.s8    q0, d0            \n"// k00-k30,k01-k31

                "vmlal.s16   q6, d4, d0[0]     \n"// sum0 = (a00-a07) * k00
                "vmlal.s16   q7, d5, d0[0]     \n"
                "vmlal.s16   q8, d4, d0[1]     \n"// sum1 = (a00-a07) * k10
                "vmlal.s16   q9, d5, d0[1]     \n"
                "vmlal.s16   q10, d4, d0[2]    \n"// sum2 = (a00-a07) * k20
                "vmlal.s16   q11, d5, d0[2]    \n"
                "vmlal.s16   q12, d4, d0[3]    \n"// sum3 = (a00-a07) * k30
                "vmlal.s16   q13, d5, d0[3]    \n"

                "vmlal.s16   q6, d6, d1[0]     \n"// sum0 += (a10-a17) * k01
                "vmlal.s16   q7, d7, d1[0]     \n"
                "vmlal.s16   q8, d6, d1[1]     \n"// sum1 += (a10-a17) * k11
                "vmlal.s16   q9, d7, d1[1]     \n"
                "vmlal.s16   q10, d6, d1[2]    \n"// sum2 += (a10-a17) * k21
                "vmlal.s16   q11, d7, d1[2]    \n"
                "vmlal.s16   q12, d6, d1[3]    \n"// sum3 += (a10-a17) * k31
                "vmlal.s16   q13, d7, d1[3]    \n"

                "vmlal.s16   q6, d8, d2[0]     \n"// sum0 += (a20-a27) * k02
                "vmlal.s16   q7, d9, d2[0]     \n"
                "vmlal.s16   q8, d8, d2[1]     \n"// sum1 += (a20-a27) * k12
                "vmlal.s16   q9, d9, d2[1]     \n"
                "vmlal.s16   q10, d8, d2[2]    \n"// sum2 += (a20-a27) * k22
                "vmlal.s16   q11, d9, d2[2]    \n"
                "vmlal.s16   q12, d8, d2[3]    \n"// sum3 += (a20-a27) * k32
                "vmlal.s16   q13, d9, d2[3]    \n"  

                "vmlal.s16   q6, d10, d3[0]    \n"// sum0 += (a30-a37) * k03
                "vmlal.s16   q7, d11, d3[0]    \n"
                "vmlal.s16   q8, d10, d3[1]    \n"// sum1 += (a30-a37) * k13
                "vmlal.s16   q9, d11, d3[1]    \n"
                "vmlal.s16   q10, d10, d3[2]   \n"// sum2 += (a30-a37) * k23
                "vmlal.s16   q11, d11, d3[2]   \n"
                "vmlal.s16   q12, d10, d3[3]   \n"// sum3 += (a30-a37) * k33
                "vmlal.s16   q13, d11, d3[3]   \n"                  

                "subs        r4, r4, #1        \n"
                "bne         0b                \n"// end for
 
                "1:                            \n"
                // remain loop
                "and         r4, %12, #3       \n"// r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n"// for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%4]!       \n"// tmpr a00-a07    a(inch)(data)
                "vld1.s8     {d0}, [%5]        \n"// kptr k00-k30    k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %5, #4            \n"

                "vmlal.s16   q6, d2, d0[0]     \n"// sum0 += (a00-a07) * k00
                "vmlal.s16   q7, d3, d0[0]     \n"
                "vmlal.s16   q8, d2, d0[1]     \n"// sum1 += (a00-a07) * k10
                "vmlal.s16   q9, d3, d0[1]     \n"
                "vmlal.s16   q10, d2, d0[2]    \n"// sum2 += (a00-a07) * k20
                "vmlal.s16   q11, d3, d0[2]    \n"
                "vmlal.s16   q12, d2, d0[3]    \n"// sum3 += (a00-a07) * k30
                "vmlal.s16   q13, d3, d0[3]    \n"    

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                            \n"// store the result to memory
                "vst1.s32    {d12-d15}, [%0]!  \n"
                "vst1.s32    {d16-d19}, [%1]!  \n"
                "vst1.s32    {d20-d23}, [%2]!  \n"
                "vst1.s32    {d24-d27}, [%3]!  \n"

                : "=r"(outptr0), // %0
                  "=r"(outptr1), // %1
                  "=r"(outptr2), // %2
                  "=r"(outptr3), // %3
                  "=r"(tmpptr),  // %4
                  "=r"(kptr)     // %5
                : "0"(outptr0),
                  "1"(outptr1),
                  "2"(outptr2),
                  "3"(outptr3),
                  "4"(tmpptr),
                  "5"(kptr),
                  "r"(inch)      // %12  
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
            );
#endif // __aarch64__            
#else
           	short sum0_0 = 0;
            short sum0_1 = 0;
            short sum0_2 = 0;
            short sum0_3 = 0;
            short sum0_4 = 0;
            short sum0_5 = 0;
            short sum0_6 = 0;
            short sum0_7 = 0;

            short sum1_0 = 0;
            short sum1_1 = 0;
            short sum1_2 = 0;
            short sum1_3 = 0;
            short sum1_4 = 0;
            short sum1_5 = 0;
            short sum1_6 = 0;
            short sum1_7 = 0;

            short sum2_0 = 0;
            short sum2_1 = 0;
            short sum2_2 = 0;
            short sum2_3 = 0;
            short sum2_4 = 0;
            short sum2_5 = 0;
            short sum2_6 = 0;
            short sum2_7 = 0;

            short sum3_0 = 0;
            short sum3_1 = 0;
            short sum3_2 = 0;
            short sum3_3 = 0;
            short sum3_4 = 0;
            short sum3_5 = 0;
            short sum3_6 = 0;
            short sum3_7 = 0;

            for (int q=0; q<inch; q++)
            {
                sum0_0 = saturate2int16((int)sum0_0 + ((((short)tmpptr[0] * kptr[0]) + 1) >> 1));
                sum0_1 = saturate2int16((int)sum0_1 + ((((short)tmpptr[1] * kptr[0]) + 1) >> 1));
                sum0_2 = saturate2int16((int)sum0_2 + ((((short)tmpptr[2] * kptr[0]) + 1) >> 1));
                sum0_3 = saturate2int16((int)sum0_3 + ((((short)tmpptr[3] * kptr[0]) + 1) >> 1));
                sum0_4 = saturate2int16((int)sum0_4 + ((((short)tmpptr[4] * kptr[0]) + 1) >> 1));
                sum0_5 = saturate2int16((int)sum0_5 + ((((short)tmpptr[5] * kptr[0]) + 1) >> 1));
                sum0_6 = saturate2int16((int)sum0_6 + ((((short)tmpptr[6] * kptr[0]) + 1) >> 1));
                sum0_7 = saturate2int16((int)sum0_7 + ((((short)tmpptr[7] * kptr[0]) + 1) >> 1));

                sum1_0 = saturate2int16((int)sum1_0 + ((((short)tmpptr[0] * kptr[1]) + 1) >> 1));
                sum1_1 = saturate2int16((int)sum1_1 + ((((short)tmpptr[1] * kptr[1]) + 1) >> 1));
                sum1_2 = saturate2int16((int)sum1_2 + ((((short)tmpptr[2] * kptr[1]) + 1) >> 1));
                sum1_3 = saturate2int16((int)sum1_3 + ((((short)tmpptr[3] * kptr[1]) + 1) >> 1));
                sum1_4 = saturate2int16((int)sum1_4 + ((((short)tmpptr[4] * kptr[1]) + 1) >> 1));
                sum1_5 = saturate2int16((int)sum1_5 + ((((short)tmpptr[5] * kptr[1]) + 1) >> 1));
                sum1_6 = saturate2int16((int)sum1_6 + ((((short)tmpptr[6] * kptr[1]) + 1) >> 1));
                sum1_7 = saturate2int16((int)sum1_7 + ((((short)tmpptr[7] * kptr[1]) + 1) >> 1));

                sum2_0 = saturate2int16((int)sum2_0 + ((((short)tmpptr[0] * kptr[2]) + 1) >> 1));
                sum2_1 = saturate2int16((int)sum2_1 + ((((short)tmpptr[1] * kptr[2]) + 1) >> 1));
                sum2_2 = saturate2int16((int)sum2_2 + ((((short)tmpptr[2] * kptr[2]) + 1) >> 1));
                sum2_3 = saturate2int16((int)sum2_3 + ((((short)tmpptr[3] * kptr[2]) + 1) >> 1));
                sum2_4 = saturate2int16((int)sum2_4 + ((((short)tmpptr[4] * kptr[2]) + 1) >> 1));
                sum2_5 = saturate2int16((int)sum2_5 + ((((short)tmpptr[5] * kptr[2]) + 1) >> 1));
                sum2_6 = saturate2int16((int)sum2_6 + ((((short)tmpptr[6] * kptr[2]) + 1) >> 1));
                sum2_7 = saturate2int16((int)sum2_7 + ((((short)tmpptr[7] * kptr[2]) + 1) >> 1));

                sum3_0 = saturate2int16((int)sum3_0 + ((((short)tmpptr[0] * kptr[3]) + 1) >> 1));
                sum3_1 = saturate2int16((int)sum3_1 + ((((short)tmpptr[1] * kptr[3]) + 1) >> 1));
                sum3_2 = saturate2int16((int)sum3_2 + ((((short)tmpptr[2] * kptr[3]) + 1) >> 1));
                sum3_3 = saturate2int16((int)sum3_3 + ((((short)tmpptr[3] * kptr[3]) + 1) >> 1));
                sum3_4 = saturate2int16((int)sum3_4 + ((((short)tmpptr[4] * kptr[3]) + 1) >> 1));
                sum3_5 = saturate2int16((int)sum3_5 + ((((short)tmpptr[5] * kptr[3]) + 1) >> 1));
                sum3_6 = saturate2int16((int)sum3_6 + ((((short)tmpptr[6] * kptr[3]) + 1) >> 1));
                sum3_7 = saturate2int16((int)sum3_7 + ((((short)tmpptr[7] * kptr[3]) + 1) >> 1));

                tmpptr += 8;
                kptr += 4;
            }

            outptr0[0] = sum0_0;
            outptr0[1] = sum0_1;
            outptr0[2] = sum0_2;
            outptr0[3] = sum0_3;
            outptr0[4] = sum0_4;
            outptr0[5] = sum0_5;
            outptr0[6] = sum0_6;
            outptr0[7] = sum0_7;

            outptr1[0] = sum1_0;
            outptr1[1] = sum1_1;
            outptr1[2] = sum1_2;
            outptr1[3] = sum1_3;
            outptr1[4] = sum1_4;
            outptr1[5] = sum1_5;
            outptr1[6] = sum1_6;
            outptr1[7] = sum1_7;

            outptr2[0] = sum2_0;
            outptr2[1] = sum2_1;
            outptr2[2] = sum2_2;
            outptr2[3] = sum2_3;
            outptr2[4] = sum2_4;
            outptr2[5] = sum2_5;
            outptr2[6] = sum2_6;
            outptr2[7] = sum2_7;

            outptr3[0] = sum3_0;
            outptr3[1] = sum3_1;
            outptr3[2] = sum3_2;
            outptr3[3] = sum3_3;
            outptr3[4] = sum3_4;
            outptr3[5] = sum3_5;
            outptr3[6] = sum3_6;
            outptr3[7] = sum3_7;

            outptr0 += 8;
            outptr1 += 8;
            outptr2 += 8;
            outptr3 += 8;
#endif // __ARM_NEON            
        }    

        for (; i<size; i++)
        {
            const signed char* tmpptr = tmp.channel(i/8 + i%8);
#if 1 //__ARM_NEON && __aarch64__
            const signed char* kptr = kernel.channel(p/8 + (p%8)/4);
#else
            const signed char* kptr = kernel.channel(p/4);
#endif // __ARM_NEON && __aarch64__
#if 1 //__ARM_NEON
#if __aarch64__
            asm volatile(
                "eor    v14.16b, v14.16b, v14.16b    \n" // sum0_3
                "eor    v16.16b, v16.16b, v16.16b    \n" // sum0
                "eor    v17.16b, v17.16b, v17.16b    \n" // sum1
                "eor    v18.16b, v18.16b, v18.16b    \n" // sum2
                "eor    v19.16b, v19.16b, v19.16b    \n" // sum3

                // inch loop
                "lsr    w4, %w12, #2                 \n"// w4 = nn = inch >> 2
                "cmp    w4, #0                       \n"
                "beq    1f                           \n"

                "0:                                  \n"
                "ld1    {v0.8b, v1.8b}, [%5], #16    \n"
                "ld1    {v8.8b}, [%4]                \n"
                "add    %4, %4, #4                   \n"

                "ext      v2.8b, v0.8b, v0.8b, #4    \n" // k01-k31
                "ext      v3.8b, v1.8b, v1.8b, #4    \n" // k03-k33

                "dup      v9.8b, v8.b[0]             \n" // a00
                "dup      v10.8b, v8.b[1]            \n" // a10
                "dup      v11.8b, v8.b[2]            \n" // a20
                "dup      v12.8b, v8.b[3]            \n" // a30                

                "subs   w4, w4, #1                   \n"

                //
                "smull    v17.8h, v0.8b, v9.8b       \n"// sum0 += (k00-k30) * a00
                "smull    v18.8h, v2.8b, v10.8b      \n"// sum1 += (k01-k31) * a10
                "smull    v19.8h, v1.8b, v11.8b      \n"// sum2 += (k02-k32) * a20
                "smull    v20.8h, v3.8b, v12.8b      \n"// sum3 += (k03-k33) * a30

                "add      v17.8h, v17.8h, v18.8h     \n"
                "add      v20.8h, v20.8h, v19.8h     \n"
                "add      v21.8h, v17.8h, v20.8h     \n"

                // "add      v21.8h, v21.8h, %13.8h     \n"
                // "sshr     v21.8h, v21.8h, #1         \n"

                "sqadd    v22.8h, v22.8h, v21.8h     \n"                

                "bne    0b                           \n"

                "1:                                  \n"

                // remain loop
                "and    w4, %w12, #3                 \n"// w4 = remain = inch & 3;
                "cmp    w4, #0                       \n"
                "beq    3f                           \n"

                "2:                                  \n"

                //"prfm   pldl1keep, [%5, #128]      \n"
                "ld1    {v0.8b}, [%5]                \n"// k
                "ld1    {v8.8b}, [%4]                \n"// d
                "add    %4, %4, #1                   \n"
                "add    %5, %5, #4                   \n"

                "dup    v9.8b, v8.b[0]               \n" // a00

                "subs   w4, w4, #1                   \n"

                // k0
                "smull    v17.8h, v0.8b, v9.8b       \n"// sum0 += (k00-k30) * a00

                // "add      v17.8h, v17.8h, %13.8h     \n"
                // "sshr     v17.8h, v17.8h, #1         \n"

                "sqadd    v22.8h, v22.8h, v17.8h     \n"

                "bne    2b                           \n"

                "3:                                  \n"

                "st1    {v22.h}[0], [%0], #2         \n"
                "st1    {v22.h}[1], [%1], #2         \n"
                "st1    {v22.h}[2], [%2], #2         \n"
                "st1    {v22.h}[3], [%3], #2         \n"

                : "=r"(outptr0),    // %0
                  "=r"(outptr1),    // %1
                  "=r"(outptr2),    // %2
                  "=r"(outptr3),    // %3
                  "=r"(tmpptr),     // %4
                  "=r"(kptr)        // %5
                : "0"(outptr0),
                  "1"(outptr1),
                  "2"(outptr2),
                  "3"(outptr3),
                  "4"(tmpptr),
                  "5"(kptr),
                  "r"(inch),        // %12
                  "w"(_int1)        // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19"
            );
#else
            asm volatile(
                // inch loop
                "veor        q6, q6, q6        \n"
                "veor        q7, q7, q7        \n"
                "veor        q8, q8, q8        \n"
                "veor        q9, q9, q9        \n"
                "vmov.s32    q10, #0           \n"

                "lsr         r4, %12, #2       \n"// r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"
                
                "0:                            \n"// for(; nn != 0; nn--)
                "pld         [%4, #128]        \n"
                "vld1.s8     {d4}, [%4]        \n"// tmpr a00,a10,a20,a30    a(inch)(data)
                "add         %4, #4            \n"
                "vmovl.s8    q2, d4            \n"// a00,a10,a20,a30

                "vld1.s8     {d0-d1}, [%5]!    \n"// kptr k00-k30,k01-k31,k02-k32,k03-k33    k(outch)(inch)
                "vmovl.s8    q1, d1            \n"// k02-k32,k03-k33
                "vmovl.s8    q0, d0            \n"// k00-k30,k01-k31

                "vmlal.s16   q6, d0, d4[0]     \n"// (k00-k30) * a00
                "vmlal.s16   q7, d1, d4[1]     \n"// (k01-k31) * a10
                "vmlal.s16   q8, d2, d4[2]     \n"// (k02-k32) * a20
                "vmlal.s16   q9, d3, d4[3]     \n"// (k03-k33) * a30

                "subs        r4, r4, #1        \n"
                "bne         0b                \n"// end for

                "vadd.s32    q6, q6, q7        \n"
                "vadd.s32    q9, q9, q8        \n"
                "vadd.s32    q10, q6, q9       \n"
 
                "1:                            \n"
                // remain loop
                "and         r4, %12, #3       \n"// r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n"// for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%4]        \n"// tmpr a00        a(inch)(data)
                "vld1.s8     {d0}, [%5]        \n"// kptr k00-k30    k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %4, #1            \n"
                "add         %5, #4            \n"

                "vmlal.s16   q10, d0, d2[0]    \n"

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                            \n"// store the result to memory
                "vst1.s32    {d20[0]}, [%0]!   \n"
                "vst1.s32    {d20[1]}, [%1]!   \n"
                "vst1.s32    {d21[0]}, [%2]!   \n"
                "vst1.s32    {d21[1]}, [%3]!   \n"

                : "=r"(outptr0), // %0
                  "=r"(outptr1), // %1
                  "=r"(outptr2), // %2
                  "=r"(outptr3), // %3
                  "=r"(tmpptr),  // %4
                  "=r"(kptr)     // %5
                : "0"(outptr0),
                  "1"(outptr1),
                  "2"(outptr2),
                  "3"(outptr3),
                  "4"(tmpptr),
                  "5"(kptr),
                  "r"(inch)      // %12  
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
            );
#endif // __aarch64__            
#else
            short sum0 = 0;
            short sum1 = 0;
            short sum2 = 0;
            short sum3 = 0;

            for (int q=0; q<inch; q++)
            {
                sum0 = saturate2int16((int)sum0 + ((((short)tmpptr[0] * kptr[0]) + 1) >> 1));
                sum1 = saturate2int16((int)sum1 + ((((short)tmpptr[0] * kptr[1]) + 1) >> 1));
                sum2 = saturate2int16((int)sum2 + ((((short)tmpptr[0] * kptr[2]) + 1) >> 1));
                sum3 = saturate2int16((int)sum3 + ((((short)tmpptr[0] * kptr[3]) + 1) >> 1));

                tmpptr++;
                kptr += 4;
            }

            outptr0[0] = sum0;
            outptr1[0] = sum1;
            outptr2[0] = sum2;
            outptr3[0] = sum3;

            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;  
#endif // __ARM_NEON
        }
    }

    remain_outch_start += nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        short* outptr0 = out0;

        int i = 0;

        for (; i+7<size; i+=8)
        {
            const signed char* tmpptr = tmp.channel(i/8);
#if 1 //__ARM_NEON && __aarch64__
            const signed char* kptr = kernel.channel(p/8 + (p%8)/4 + p%4);
#else
            const signed char* kptr = kernel.channel(p/4 + p%4);
#endif // __ARM_NEON && __aarch64__
#if 1 //__ARM_NEON
#if __aarch64__
            asm volatile(
                "eor    v16.16b, v16.16b, v16.16b   \n" // sum0

                // inch loop
                "lsr    w4, %w6, #2                 \n"// w4 = nn = inch >> 2
                "cmp    w4, #0                      \n"
                "beq    1f                          \n"

                "0:                                 \n"
                "ld1    {v0.8b}, [%2]               \n"
                "ld1    {v8.8b, v9.8b, v10.8b, v11.8b}, [%1], #32   \n"
                "add    %2, %2, #4                   \n"

                "dup      v4.8b, v0.b[0]            \n"// k00 - k30
                "dup      v5.8b, v0.b[1]            \n"
                "dup      v6.8b, v0.b[2]            \n"
                "dup      v7.8b, v0.b[3]            \n"

                // k0
                "smull    v12.8h, v8.8b, v4.8b      \n"// sum0 += (a00-a70) * k00
                // k1
                "smlal    v12.8h, v9.8b, v5.8b      \n"// sum0 += (a01-a71) * k01
                // k2
                "smlal    v12.8h, v10.8b, v6.8b     \n"// sum0 += (a20-a27) * k02
                "subs   w4, w4, #1                  \n"
                // k3
                "smlal    v12.8h, v11.8b, v7.8b     \n"// sum0 += (a30-a37) * k03

                // "add       v12.8h, v12.8h, %7.8h    \n"
                // "sshr      v12.8h, v12.8h, #1       \n"

                "sqadd     v16.8h, v16.8h, v12.8h   \n"

                "bne    0b                          \n"

                "1:                                 \n"

                // remain loop
                "and    w4, %w6, #3                 \n"// w4 = remain = inch & 3;
                "cmp    w4, #0                      \n"
                "beq    3f                          \n"

                "2:                                 \n"
                "ld1    {v0.8b}, [%2]               \n"
                "ld1    {v8.8b}, [%1], #8           \n"
                "add    %2, %2, #1                  \n"

                "dup    v4.8b, v0.b[0]              \n"// k00

                "subs   w4, w4, #1                  \n"

                // k0
                "smull    v12.8h, v8.8b, v4.8b      \n"// sum0 += (a00-a70) * k00

                // "add      v12.8h, v12.8h, %7.8h     \n"
                // "sshr     v12.8h, v12.8h, #1        \n"

                "sqadd    v16.8h, v16.8h, v12.8h    \n"

                "bne    2b                          \n"

                "3:                                 \n"

                "st1    {v16.8h}, [%0], #16         \n"

                : "=r"(outptr0),    // %0
                  "=r"(tmpptr),     // %1
                  "=r"(kptr)        // %2
                : "0"(outptr0),
                  "1"(tmpptr),
                  "2"(kptr),
                  "r"(inch),        // %6
                  "w"(_int1)        // %7
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"
            );
#else
            asm volatile(
                // inch loop
                "vmov.s32    q6, #0            \n"
                "vmov.s32    q7, #0            \n"

                "lsr         r4, %6, #2        \n"// r4 = nn = inch >> 2
                "cmp         r4, #0            \n"
                "beq         1f                \n"
                
                "0:                            \n"// for(; nn != 0; nn--)
                "pld         [%1, #128]        \n"
                "vld1.s8     {d4-d7}, [%1]!    \n"// tmpr a00-a07,a10-a17,a20-a27,a30-a37    a(inch)(data)
                "vmovl.s8    q5, d7            \n"// a30-a37
                "vmovl.s8    q4, d6            \n"// a20-a27
                "vmovl.s8    q3, d5            \n"// a10-a17
                "vmovl.s8    q2, d4            \n"// a00-a07

                "vld1.s8     {d0}, [%2]        \n"// kptr k00,k01,k02,k03    k(outch)(inch)
                "vmovl.s8    q0, d0            \n"// k00,k01,k02,k03
                "add         %2, #4            \n"

                "vmlal.s16   q6, d4, d0[0]     \n"// (a00-a07) * k00
                "vmlal.s16   q7, d5, d0[0]     \n"
                "vmlal.s16   q6, d6, d0[1]     \n"// (a10-a17) * k01
                "vmlal.s16   q7, d7, d0[1]     \n"
                "vmlal.s16   q6, d8, d0[2]     \n"// (a20-a27) * k02
                "vmlal.s16   q7, d9, d0[2]     \n"
                "vmlal.s16   q6, d10, d0[3]    \n"// (a30-a37) * k03
                "vmlal.s16   q7, d11, d0[3]    \n"

                "subs        r4, r4, #1        \n"
                "bne         0b                \n"// end for
 
                "1:                            \n"
                // remain loop
                "and         r4, %6, #3        \n"// r4 = remain = inch & 3
                "cmp         r4, #0            \n"
                "beq         3f                \n"

                "2:                            \n"// for(; remain != 0; remain--)
                "vld1.s8     {d2}, [%1]!       \n"// tmpr a00-a07    a(inch)(data)
                "vld1.s8     {d0}, [%2]        \n"// kptr k00        k(outch)(inch)
                "vmovl.s8    q1, d2            \n"
                "vmovl.s8    q0, d0            \n"
                "add         %2, #1            \n"

                "vmlal.s16   q6, d2, d0[0]     \n"// (a00-a07) * k00
                "vmlal.s16   q7, d3, d0[0]     \n"  

                "subs        r4, r4, #1        \n"
                "bne         2b                \n"

                "3:                            \n"// store the result to memory
                "vst1.s32    {d12-d15}, [%0]!  \n"

                : "=r"(outptr0), // %0
                  "=r"(tmpptr),  // %1
                  "=r"(kptr)     // %2
                : "0"(outptr0),
                  "1"(tmpptr),
                  "2"(kptr),
                  "r"(inch)      // %6  
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
            );
#endif // __aarch64__            
#else
            short sum0 = 0;
            short sum1 = 0;
            short sum2 = 0;
            short sum3 = 0;
            short sum4 = 0;
            short sum5 = 0;
            short sum6 = 0;
            short sum7 = 0;

            for (int q=0; q<inch; q++)
            {
                sum0 = saturate2int16((int)sum0 + ((((short)tmpptr[0] * kptr[0]) + 1) >> 1));
                sum1 = saturate2int16((int)sum1 + ((((short)tmpptr[1] * kptr[0]) + 1) >> 1));
                sum2 = saturate2int16((int)sum2 + ((((short)tmpptr[2] * kptr[0]) + 1) >> 1));
                sum3 = saturate2int16((int)sum3 + ((((short)tmpptr[3] * kptr[0]) + 1) >> 1));
                sum4 = saturate2int16((int)sum4 + ((((short)tmpptr[4] * kptr[0]) + 1) >> 1));
                sum5 = saturate2int16((int)sum5 + ((((short)tmpptr[5] * kptr[0]) + 1) >> 1));
                sum6 = saturate2int16((int)sum6 + ((((short)tmpptr[6] * kptr[0]) + 1) >> 1));
                sum7 = saturate2int16((int)sum7 + ((((short)tmpptr[7] * kptr[0]) + 1) >> 1));

                tmpptr += 8;
                kptr++;
            }

            outptr0[0] = sum0;
            outptr0[1] = sum1;
            outptr0[2] = sum2;
            outptr0[3] = sum3;
            outptr0[4] = sum4;
            outptr0[5] = sum5;
            outptr0[6] = sum6;
            outptr0[7] = sum7;

            outptr0 += 8;
#endif // __ARM_NEON
        }   

        for (; i<size; i++)
        {
            const signed char* tmpptr = tmp.channel(i/8 + i%8);   
#if 1 // __ARM_NEON && __aarch64__
            const signed char* kptr = kernel.channel(p/8 + (p%8)/4 + p%4);
#else
            const signed char* kptr = kernel.channel(p/4 + p%4);
#endif // __ARM_NEON && __aarch64__

            int q = 0;            
            short sum0 = 0;

            for (; q<inch; q++)
            {
                // sum0 = saturate2int16((int)sum0 + ((((short)tmpptr[0] * kptr[0]) + 1) >> 1));
                sum0 = saturate2int16((int)sum0 + ((short)tmpptr[0] * kptr[0]));
                tmpptr++;
                kptr++;
            }

            outptr0[0] = sum0;

            outptr0++;
        }
    }   

//     // NOTE sgemm int8
//     for (; p<outch; p++)
//     {
//         Mat out0 = top_blob.channel(p);
//
//         int* outptr0 = out0;
//
//         for (int i=0; i<size; i++)
//         {
//             int sum = 0;
//
//             const signed char* kptr = _kernel.channel(p/8 + p%8);
//
//             for (int q=0; q<inch; q++)
//             {
//                 const signed char* img0 = bottom_blob.channel(q);
//
//                 sum += img0[i] * kptr[0];
//                 kptr ++;
//             }
//
//             outptr0[i] = sum;
//         }
//     }
}

/*
 * Convolution 1x1 quantized with int8,unroll 16 x 8,step size 16
 */
static void conv1x1s1_int8_e2e_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const signed char* kernel = _kernel;

    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 3;
    remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 8;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p+1);
        Mat out2 = top_blob.channel(p+2);
        Mat out3 = top_blob.channel(p+3);
        Mat out4 = top_blob.channel(p+4);
        Mat out5 = top_blob.channel(p+5);
        Mat out6 = top_blob.channel(p+6);
        Mat out7 = top_blob.channel(p+7);

        out0.fill(0);
        out1.fill(0);
        out2.fill(0);
        out3.fill(0);
        out4.fill(0);
        out5.fill(0);
        out6.fill(0);
        out7.fill(0);

        int q = 0;

#ifdef __clang__
        for (; q+15<inch; q+=16)
        {
            short* outptr0 = out0;
            short* outptr1 = out1;
            short* outptr2 = out2;
            short* outptr3 = out3;
            short* outptr4 = out4;
            short* outptr5 = out5;
            short* outptr6 = out6;
            short* outptr7 = out7;

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;
            const signed char* kernel1 = (const signed char*)kernel + (p+1)*inch + q;
            const signed char* kernel2 = (const signed char*)kernel + (p+2)*inch + q;
            const signed char* kernel3 = (const signed char*)kernel + (p+3)*inch + q;
            const signed char* kernel4 = (const signed char*)kernel + (p+4)*inch + q;
            const signed char* kernel5 = (const signed char*)kernel + (p+5)*inch + q;
            const signed char* kernel6 = (const signed char*)kernel + (p+6)*inch + q;
            const signed char* kernel7 = (const signed char*)kernel + (p+7)*inch + q;

            const signed char* r0 = bottom_blob.channel(q);
            const signed char* r1 = bottom_blob.channel(q+1);
            const signed char* r2 = bottom_blob.channel(q+2);
            const signed char* r3 = bottom_blob.channel(q+3);
            const signed char* r4 = bottom_blob.channel(q+4);
            const signed char* r5 = bottom_blob.channel(q+5);
            const signed char* r6 = bottom_blob.channel(q+6);
            const signed char* r7 = bottom_blob.channel(q+7);
            const signed char* r8 = bottom_blob.channel(q+8);
            const signed char* r9 = bottom_blob.channel(q+9);
            const signed char* r10 = bottom_blob.channel(q+10);
            const signed char* r11 = bottom_blob.channel(q+11);
            const signed char* r12 = bottom_blob.channel(q+12);
            const signed char* r13 = bottom_blob.channel(q+13);
            const signed char* r14 = bottom_blob.channel(q+14);
            const signed char* r15 = bottom_blob.channel(q+15);

            int size = outw * outh;

            int nn = size >> 4;
            int remain = size & 15;

            int8x16_t _k0 = vld1q_s8(kernel0);
            int8x16_t _k1 = vld1q_s8(kernel1);
            int8x16_t _k2 = vld1q_s8(kernel2);
            int8x16_t _k3 = vld1q_s8(kernel3);
            int8x16_t _k4 = vld1q_s8(kernel4);
            int8x16_t _k5 = vld1q_s8(kernel5);
            int8x16_t _k6 = vld1q_s8(kernel6);
            int8x16_t _k7 = vld1q_s8(kernel7);

            if (nn > 0)
            {
            asm volatile(
                "prfm   pldl1keep, [%9, #128]        \n"
                "prfm   pldl1keep, [%10, #128]       \n"
                "prfm   pldl1keep, [%11, #128]       \n"
                "prfm   pldl1keep, [%12, #128]       \n"
                "ld1    {v8.16b}, [%9], #16          \n" // r0"
                "ld1    {v9.16b}, [%10], #16         \n" // r1"
                "ld1    {v10.16b}, [%11], #16        \n" // r2"
                "ld1    {v11.16b}, [%12], #16        \n" // r3"

                "dup    v24.16b, %50.b[0]            \n" // k00
                "dup    v25.16b, %50.b[1]            \n" // k01
                "dup    v26.16b, %50.b[2]            \n" // k02
                "dup    v27.16b, %50.b[3]            \n" // k03

                "0:                                  \n"
                "smull  v28.8h, v8.8b, v24.8b        \n" // r0 * k0
                "smull2  v31.8h, v8.16b, v24.16b     \n" // r0n * k0
                "prfm   pldl1keep, [%13, #128]       \n"
                "prfm   pldl1keep, [%14, #128]       \n"
                "prfm   pldl1keep, [%15, #128]       \n"

                "smlal  v28.8h, v9.8b, v25.8b        \n" // r0 * k1
                "smlal2  v31.8h, v9.16b, v25.16b     \n" // r0n * k1
                "prfm   pldl1keep, [%16, #128]       \n"
                "ld1    {v12.16b}, [%13], #16        \n" // r4"   
                "ld1    {v13.16b}, [%14], #16        \n" // r5"

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "smlal2  v31.8h, v10.16b, v26.16b    \n"
                "ld1    {v14.16b}, [%15], #16        \n" // r6"
                "ld1    {v15.16b}, [%16], #16        \n" // r7"                               
                "dup    v24.16b, %50.b[4]            \n" // k04

                "smlal  v28.8h, v11.8b, v27.8b       \n"
                "smlal2  v31.8h, v11.16b, v27.16b    \n"
                "dup    v25.16b, %50.b[5]            \n" // k05
                "dup    v26.16b, %50.b[6]            \n" // k06
                "dup    v27.16b, %50.b[7]            \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n" // r4
                "smlal2  v31.8h, v12.16b, v24.16b    \n" // r4
                "prfm   pldl1keep, [%1, #128]        \n"
                "ld1    {v29.8h, v30.8h}, [%1]       \n" // sum0  
                "prfm   pldl1keep, [%17, #128]       \n"

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "smlal2  v31.8h, v13.16b, v25.16b    \n"
                "prfm   pldl1keep, [%18, #128]       \n"
                "prfm   pldl1keep, [%19, #128]       \n"
                "prfm   pldl1keep, [%20, #128]       \n"
                "ld1    {v16.16b}, [%17], #16        \n" // r8"   

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "smlal2  v31.8h, v14.16b, v26.16b    \n"
                "ld1    {v17.16b}, [%18], #16        \n" // r9"
                "ld1    {v18.16b}, [%19], #16        \n" // r10"
                "ld1    {v19.16b}, [%20], #16        \n" // r11"             

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "smlal2  v31.8h, v15.16b, v27.16b    \n"
                "dup    v24.16b, %50.b[8]            \n" // k08
                "dup    v25.16b, %50.b[9]            \n" // k09
                "dup    v26.16b, %50.b[10]           \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n" // r8
                "smlal2  v31.8h, v16.16b, v24.16b    \n" // r8
                "dup    v27.16b, %50.b[11]           \n" // k11
                "prfm   pldl1keep, [%21, #128]       \n"
                "prfm   pldl1keep, [%22, #128]       \n"

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "smlal2  v31.8h, v17.16b, v25.16b    \n"
                "prfm   pldl1keep, [%23, #128]       \n"
                "prfm   pldl1keep, [%24, #128]       \n"
                "ld1    {v20.16b}, [%21], #16        \n" // r12"

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "smlal2  v31.8h, v18.16b, v26.16b    \n"
                "ld1    {v21.16b}, [%22], #16        \n" // r13"
                "ld1    {v22.16b}, [%23], #16        \n" // r14"
                "ld1    {v23.16b}, [%24], #16        \n" // r15"              

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "smlal2  v31.8h, v19.16b, v27.16b    \n"
                "dup    v24.16b, %50.b[12]           \n" // k12
                "dup    v25.16b, %50.b[13]           \n" // k13
                "dup    v26.16b, %50.b[14]           \n" // k14

                "smlal  v28.8h, v20.8b, v24.8b       \n" // r12
                "smlal2  v31.8h, v20.16b, v24.16b    \n" // r12
                "dup    v27.16b, %50.b[15]           \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "smlal2  v31.8h, v21.16b, v25.16b    \n"
                "dup    v24.16b, %51.b[0]            \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "smlal2  v31.8h, v22.16b, v26.16b    \n"
                "dup    v25.16b, %51.b[1]            \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"
                "smlal2  v31.8h, v23.16b, v27.16b    \n"             
                "dup    v26.16b, %51.b[2]            \n" // k02

                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "sqadd  v30.8h, v30.8h, v31.8h       \n"
                "dup    v27.16b, %51.b[3]            \n" // k03

                //"st1    {v29.8h}, [%1], #16  		 \n" // sum0

                //"ld1    {v29.8h}, [%1]       		 \n" // sum0
                

                "st1    {v29.8h, v30.8h}, [%1], #32  \n" // sum0            
                //########################################### 
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "smull2  v31.8h, v8.16b, v24.16b     \n"
                "dup    v24.16b, %51.b[4]            \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "smlal2  v31.8h, v9.16b, v25.16b     \n"
                "dup    v25.16b, %51.b[5]            \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "smlal2  v31.8h, v10.16b, v26.16b    \n"
                "dup    v26.16b, %51.b[6]            \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"    
                "smlal2  v31.8h, v11.16b, v27.16b    \n"         
                "dup    v27.16b, %51.b[7]            \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "smlal2  v31.8h, v12.16b, v24.16b    \n"
                "prfm   pldl1keep, [%2, #128]        \n"
                "ld1    {v29.8h, v30.8h}, [%2]       \n" // sum1

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "smlal2  v31.8h, v13.16b, v25.16b    \n"
                "dup    v24.16b, %51.b[8]            \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "smlal2  v31.8h, v14.16b, v26.16b    \n"
                "dup    v25.16b, %51.b[9]            \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "smlal2  v31.8h, v15.16b, v27.16b    \n"
                "dup    v26.16b, %51.b[10]           \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "smlal2  v31.8h, v16.16b, v24.16b    \n"
                "dup    v27.16b, %51.b[11]           \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "smlal2  v31.8h, v17.16b, v25.16b    \n"
                "dup    v24.16b, %51.b[12]           \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "smlal2  v31.8h, v18.16b, v26.16b    \n"
                "dup    v25.16b, %51.b[13]           \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "smlal2  v31.8h, v19.16b, v27.16b    \n"
                "dup    v26.16b, %51.b[14]           \n" // k14

                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "smlal2  v31.8h, v20.16b, v24.16b    \n"
                "dup    v27.16b, %51.b[15]           \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "smlal2  v31.8h, v21.16b, v25.16b    \n"
                "dup    v24.16b, %52.b[0]            \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "smlal2  v31.8h, v22.16b, v26.16b    \n"
                "dup    v25.16b, %52.b[1]            \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"        
                "smlal2  v31.8h, v23.16b, v27.16b    \n"     
            
                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                
                "dup    v26.16b, %52.b[2]            \n" // k02
                "dup    v27.16b, %52.b[3]            \n" // k03  

                //"st1    {v29.8h}, [%2], #16  		 \n"

                //"ld1    {v29.8h}, [%2]       	     \n" // sum1
                "sqadd  v30.8h, v30.8h, v31.8h       \n"
                   
                "st1    {v29.8h, v30.8h}, [%2], #32  \n"             
                //########################################### // sum1

                "smull  v28.8h, v8.8b, v24.8b        \n"
                "smull2  v31.8h, v8.16b, v24.16b     \n"
                "dup    v24.16b, %52.b[4]            \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "smlal2  v31.8h, v9.16b, v25.16b     \n"
                "dup    v25.16b, %52.b[5]            \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "smlal2  v31.8h, v10.16b, v26.16b    \n"
                "dup    v26.16b, %52.b[6]            \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"
                "smlal2  v31.8h, v11.16b, v27.16b    \n"             
                "dup    v27.16b, %52.b[7]            \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "smlal2  v31.8h, v12.16b, v24.16b    \n"
                "prfm   pldl1keep, [%3, #128]        \n"
                "ld1    {v29.8h, v30.8h}, [%3]       \n" // sum2 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "smlal2  v31.8h, v13.16b, v25.16b    \n"
                "dup    v24.16b, %52.b[8]            \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "smlal2  v31.8h, v14.16b, v26.16b    \n"
                "dup    v25.16b, %52.b[9]            \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "smlal2  v31.8h, v15.16b, v27.16b    \n"
                "dup    v26.16b, %52.b[10]           \n" // k10

                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "smlal2  v31.8h, v16.16b, v24.16b    \n"
                "dup    v27.16b, %52.b[11]           \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "smlal2  v31.8h, v17.16b, v25.16b    \n"
                "dup    v24.16b, %52.b[12]           \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "smlal2  v31.8h, v18.16b, v26.16b    \n"
                "dup    v25.16b, %52.b[13]           \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "smlal2  v31.8h, v19.16b, v27.16b    \n"
                "dup    v26.16b, %52.b[14]           \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "smlal2  v31.8h, v20.16b, v24.16b    \n"
                "dup    v27.16b, %52.b[15]           \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "smlal2  v31.8h, v21.16b, v25.16b    \n"
                "dup    v24.16b, %53.b[0]            \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "smlal2  v31.8h, v22.16b, v26.16b    \n"
                "dup    v25.16b, %53.b[1]            \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"
                "smlal2  v31.8h, v23.16b, v27.16b    \n"             
            
                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "dup    v26.16b, %53.b[2]            \n" // k02

                
                "dup    v27.16b, %53.b[3]            \n" // k03

                //"st1    {v29.8h}, [%3], #16  		 \n"
                //"ld1    {v29.8h}, [%3]       		 \n" // sum2 
                "sqadd  v30.8h, v30.8h, v31.8h       \n"
                
                "st1    {v29.8h, v30.8h}, [%3], #32  \n"
                //########################################### //sum 2

                "smull  v28.8h, v8.8b, v24.8b        \n"
                "smull2  v31.8h, v8.16b, v24.16b     \n"
                "dup    v24.16b, %53.b[4]            \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "smlal2  v31.8h, v9.16b, v25.16b     \n"
                "dup    v25.16b, %53.b[5]            \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "smlal2  v31.8h, v10.16b, v26.16b    \n"
                "dup    v26.16b, %53.b[6]            \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"
                "smlal2  v31.8h, v11.16b, v27.16b    \n"             
                "dup    v27.16b, %53.b[7]            \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "smlal2  v31.8h, v12.16b, v24.16b    \n"
                "prfm   pldl1keep, [%4, #128]        \n"
                "ld1    {v29.8h, v30.8h}, [%4]       		 \n" // sum3 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "smlal2  v31.8h, v13.16b, v25.16b    \n"
                "dup    v24.16b, %53.b[8]            \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "smlal2  v31.8h, v14.16b, v26.16b    \n"
                "dup    v25.16b, %53.b[9]            \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "smlal2  v31.8h, v15.16b, v27.16b    \n"
                "dup    v26.16b, %53.b[10]           \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "smlal2  v31.8h, v16.16b, v24.16b    \n"
                "dup    v27.16b, %53.b[11]           \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "smlal2  v31.8h, v17.16b, v25.16b    \n"
                "dup    v24.16b, %53.b[12]           \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "smlal2  v31.8h, v18.16b, v26.16b    \n"
                "dup    v25.16b, %53.b[13]           \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "smlal2  v31.8h, v19.16b, v27.16b    \n"
                "dup    v26.16b, %53.b[14]           \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "smlal2  v31.8h, v20.16b, v24.16b    \n"
                "dup    v27.16b, %53.b[15]           \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "smlal2  v31.8h, v21.16b, v25.16b    \n"
                "dup    v24.16b, %54.b[0]            \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "smlal2  v31.8h, v22.16b, v26.16b    \n"
                "dup    v25.16b, %54.b[1]            \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"    
                "smlal2  v31.8h, v23.16b, v27.16b    \n"         
            
                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "dup    v26.16b, %54.b[2]            \n" // k02
                "dup    v27.16b, %54.b[3]            \n" // k03

                //"st1    {v29.8h}, [%4], #16  		 \n"

                //"ld1    {v29.8h}, [%4]       		 \n" // sum3 
                "sqadd  v30.8h, v30.8h, v31.8h       \n"
                
                "st1    {v29.8h, v30.8h}, [%4], #32  \n"
                //########################################### // sum3
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "smull2  v31.8h, v8.16b, v24.16b     \n"
                "dup    v24.16b, %54.b[4]            \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "smlal2  v31.8h, v9.16b, v25.16b     \n"
                "dup    v25.16b, %54.b[5]            \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "smlal2  v31.8h, v10.16b, v26.16b    \n"
                "dup    v26.16b, %54.b[6]            \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"    
                "smlal2  v31.8h, v11.16b, v27.16b    \n"         
                "dup    v27.16b, %54.b[7]            \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "smlal2  v31.8h, v12.16b, v24.16b    \n"
                "prfm   pldl1keep, [%5, #128]        \n"
                "ld1    {v29.8h, v30.8h}, [%5]       \n" // sum4

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "smlal2  v31.8h, v13.16b, v25.16b    \n"
                "dup    v24.16b, %54.b[8]            \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "smlal2  v31.8h, v14.16b, v26.16b    \n"
                "dup    v25.16b, %54.b[9]            \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"    
                "smlal2  v31.8h, v15.16b, v27.16b    \n"
                "dup    v26.16b, %54.b[10]           \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "smlal2  v31.8h, v16.16b, v24.16b    \n"
                "dup    v27.16b, %54.b[11]           \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "smlal2  v31.8h, v17.16b, v25.16b    \n"
                "dup    v24.16b, %54.b[12]           \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "smlal2  v31.8h, v18.16b, v26.16b    \n"
                "dup    v25.16b, %54.b[13]           \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "smlal2  v31.8h, v19.16b, v27.16b    \n"
                "dup    v26.16b, %54.b[14]           \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "smlal2  v31.8h, v20.16b, v24.16b    \n"
                "dup    v27.16b, %54.b[15]           \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "smlal2  v31.8h, v21.16b, v25.16b    \n"
                "dup    v24.16b, %55.b[0]            \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "smlal2  v31.8h, v22.16b, v26.16b    \n"
                "dup    v25.16b, %55.b[1]            \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"
                "smlal2  v31.8h, v23.16b, v27.16b    \n"
                "dup    v26.16b, %55.b[2]            \n" // k02    
            
                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "dup    v27.16b, %55.b[3]            \n" // k03
                //"st1    {v29.8h}, [%5], #16  		 \n"

                //"ld1    {v29.8h}, [%5]       		 \n" // sum4
                "sqadd  v30.8h, v30.8h, v31.8h       \n"
                
                "st1    {v29.8h, v30.8h}, [%5], #32  \n"
                //########################################### // sum4
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "smull2  v31.8h, v8.16b, v24.16b     \n"
                "dup    v24.16b, %55.b[4]            \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "smlal2  v31.8h, v9.16b, v25.16b     \n"
                "dup    v25.16b, %55.b[5]            \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "smlal2  v31.8h, v10.16b, v26.16b    \n"
                "dup    v26.16b, %55.b[6]            \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"
                "smlal2  v31.8h, v11.16b, v27.16b    \n"             
                "dup    v27.16b, %55.b[7]            \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "smlal2  v31.8h, v12.16b, v24.16b    \n"
                "prfm   pldl1keep, [%6, #128]        \n"
                "ld1    {v29.8h, v30.8h}, [%6]       \n" // sum5  

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "smlal2  v31.8h, v13.16b, v25.16b    \n"
                "dup    v24.16b, %55.b[8]            \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "smlal2  v31.8h, v14.16b, v26.16b    \n"
                "dup    v25.16b, %55.b[9]            \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "smlal2  v31.8h, v15.16b, v27.16b    \n"
                "dup    v26.16b, %55.b[10]           \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "smlal2  v31.8h, v16.16b, v24.16b    \n"
                "dup    v27.16b, %55.b[11]           \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "smlal2  v31.8h, v17.16b, v25.16b    \n"
                "dup    v24.16b, %55.b[12]           \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "smlal2  v31.8h, v18.16b, v26.16b    \n"
                "dup    v25.16b, %55.b[13]           \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "smlal2  v31.8h, v19.16b, v27.16b    \n"
                "dup    v26.16b, %55.b[14]           \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "smlal2  v31.8h, v20.16b, v24.16b    \n"
                "dup    v27.16b, %55.b[15]           \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "smlal2  v31.8h, v21.16b, v25.16b    \n"
                "dup    v24.16b, %56.b[0]            \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "smlal2  v31.8h, v22.16b, v26.16b    \n"
                "dup    v25.16b, %56.b[1]            \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"
                "smlal2  v31.8h, v23.16b, v27.16b    \n"             
            
                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "dup    v26.16b, %56.b[2]            \n" // k02
                "dup    v27.16b, %56.b[3]            \n" // k03

                //"st1    {v29.8h}, [%6], #16  		 \n"

                //"ld1    {v29.8h}, [%6]       		 \n" // sum5 
                "sqadd  v30.8h, v30.8h, v31.8h       \n"
                
                "st1    {v29.8h, v30.8h}, [%6], #32  \n"
                //########################################### // sum5
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "smull2  v31.8h, v8.16b, v24.16b     \n"
                "dup    v24.16b, %56.b[4]            \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "smlal2  v31.8h, v9.16b, v25.16b     \n"
                "dup    v25.16b, %56.b[5]            \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "smlal2  v31.8h, v10.16b, v26.16b    \n"
                "dup    v26.16b, %56.b[6]            \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"    
                "smlal2  v31.8h, v11.16b, v27.16b    \n"         
                "dup    v27.16b, %56.b[7]            \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "smlal2  v31.8h, v12.16b, v24.16b    \n"
                "prfm   pldl1keep, [%7, #128]        \n"
                "ld1    {v29.8h, v30.8h}, [%7]       \n" // sum6 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "smlal2  v31.8h, v13.16b, v25.16b    \n"
                "dup    v24.16b, %56.b[8]            \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "smlal2  v31.8h, v14.16b, v26.16b    \n"
                "dup    v25.16b, %56.b[9]            \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "smlal2  v31.8h, v15.16b, v27.16b    \n"     
                "dup    v26.16b, %56.b[10]           \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "smlal2  v31.8h, v16.16b, v24.16b    \n"
                "dup    v27.16b, %56.b[11]           \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "smlal2  v31.8h, v17.16b, v25.16b    \n"
                "dup    v24.16b, %56.b[12]           \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "smlal2  v31.8h, v18.16b, v26.16b    \n"
                "dup    v25.16b, %56.b[13]           \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "smlal2  v31.8h, v19.16b, v27.16b    \n"
                "dup    v26.16b, %56.b[14]           \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "smlal2  v31.8h, v20.16b, v24.16b    \n"
                "dup    v27.16b, %56.b[15]           \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "smlal2  v31.8h, v21.16b, v25.16b    \n"
                "dup    v24.16b, %57.b[0]            \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "smlal2  v31.8h, v22.16b, v26.16b    \n"
                "dup    v25.16b, %57.b[1]            \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"
                "smlal2  v31.8h, v23.16b, v27.16b    \n"             
                "dup    v26.16b, %57.b[2]            \n" // k02

                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                
                "dup    v27.16b, %57.b[3]            \n" // k03

                //"st1    {v29.8h}, [%7], #16  	     \n"
                //"ld1    {v29.8h}, [%7]       		 \n" // sum6 
                "sqadd  v30.8h, v30.8h, v31.8h       \n"
                     
                "st1    {v29.8h, v30.8h}, [%7], #32  \n"                           
                //########################################### // sum6
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "smull2  v31.8h, v8.16b, v24.16b     \n"
                "dup    v24.16b, %57.b[4]            \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "smlal2  v31.8h, v9.16b, v25.16b     \n"
                "dup    v25.16b, %57.b[5]            \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "smlal2  v31.8h, v10.16b, v26.16b    \n"
                "dup    v26.16b, %57.b[6]            \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"
                "smlal2  v31.8h, v11.16b, v27.16b    \n"             
                "dup    v27.16b, %57.b[7]            \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "smlal2  v31.8h, v12.16b, v24.16b    \n"
                "prfm   pldl1keep, [%8, #128]        \n"
                "ld1    {v29.8h, v30.8h}, [%8]       \n" // sum7 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "smlal2  v31.8h, v13.16b, v25.16b    \n"
                "dup    v24.16b, %57.b[8]            \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "smlal2  v31.8h, v14.16b, v26.16b    \n"
                "dup    v25.16b, %57.b[9]            \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "smlal2  v31.8h, v15.16b, v27.16b    \n"
                "dup    v26.16b, %57.b[10]           \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "smlal2  v31.8h, v16.16b, v24.16b    \n"
                "dup    v27.16b, %57.b[11]           \n" // k11
                
                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "smlal2  v31.8h, v17.16b, v25.16b    \n"
                "dup    v24.16b, %57.b[12]           \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "smlal2  v31.8h, v18.16b, v26.16b    \n"
                "dup    v25.16b, %57.b[13]           \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "smlal2  v31.8h, v19.16b, v27.16b    \n"
                "dup    v26.16b, %57.b[14]           \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "smlal2  v31.8h, v20.16b, v24.16b    \n"
                "dup    v27.16b, %57.b[15]           \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "smlal2  v31.8h, v21.16b, v25.16b    \n"
                "prfm   pldl1keep, [%9, #128]        \n"
                "prfm   pldl1keep, [%10, #128]       \n"
                "ld1    {v8.16b}, [%9], #16          \n" // r0"

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "smlal2  v31.8h, v22.16b, v26.16b    \n"
                "ld1    {v9.16b}, [%10], #16         \n" // r1"
                "prfm   pldl1keep, [%11, #128]       \n"
                "prfm   pldl1keep, [%12, #128]       \n"

                "smlal  v28.8h, v23.8b, v27.8b       \n"    
                "smlal2  v31.8h, v23.16b, v27.16b    \n"         
                "ld1    {v10.16b}, [%11], #16        \n" // r2"
                "ld1    {v11.16b}, [%12], #16        \n" // r3"
                "dup    v24.16b, %50.b[0]            \n" // k00                    

                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "dup    v25.16b, %50.b[1]            \n" // k01               
                //"st1    {v29.8h}, [%8], #16  	     \n"

                //"ld1    {v29.8h}, [%8]       		 \n" // sum7 
                "sqadd  v30.8h, v30.8h, v31.8h       \n"
                
                "dup    v26.16b, %50.b[2]            \n" // k02
                "dup    v27.16b, %50.b[3]            \n" // k03                 
                
                "st1    {v29.8h, v30.8h}, [%8], #32  \n"
                //########################################### // sum7
                "subs   %w0, %w0, #1                 \n"
                "bne    0b                           \n"
                "sub    %9, %9, #16                  \n"
                "sub    %10, %10, #16                \n"
                "sub    %11, %11, #16                \n"
                "sub    %12, %12, #16                \n"
                : "=r"(nn),     // %0
                  "=r"(outptr0),// %1
                  "=r"(outptr1),// %2
                  "=r"(outptr2),// %3
                  "=r"(outptr3),// %4
                  "=r"(outptr4),// %5
                  "=r"(outptr5),// %6
                  "=r"(outptr6),// %7
                  "=r"(outptr7),// %8
                  "=r"(r0),     // %9
                  "=r"(r1),     // %10
                  "=r"(r2),     // %11
                  "=r"(r3),     // %12
                  "=r"(r4),     // %13
                  "=r"(r5),     // %14
                  "=r"(r6),     // %15
                  "=r"(r7),     // %16
                  "=r"(r8),     // %17
                  "=r"(r9),     // %18
                  "=r"(r10),    // %19
                  "=r"(r11),    // %20
                  "=r"(r12),    // %21
                  "=r"(r13),    // %22
                  "=r"(r14),    // %23
                  "=r"(r15)     // %24
                : "0"(nn),
                  "1"(outptr0),
                  "2"(outptr1),
                  "3"(outptr2),
                  "4"(outptr3),
                  "5"(outptr4),
                  "6"(outptr5),
                  "7"(outptr6),
                  "8"(outptr7),
                  "9"(r0),
                  "10"(r1),
                  "11"(r2),
                  "12"(r3),
                  "13"(r4),
                  "14"(r5),
                  "15"(r6),
                  "16"(r7),
                  "17"(r8),
                  "18"(r9),
                  "19"(r10),
                  "20"(r11),
                  "21"(r12),
                  "22"(r13),
                  "23"(r14),
                  "24"(r15),
                  "w"(_k0),     // %50
                  "w"(_k1),     // %51
                  "w"(_k2),     // %52
                  "w"(_k3),     // %53
                  "w"(_k4),     // %54
                  "w"(_k5),     // %55
                  "w"(_k6),     // %56
                  "w"(_k7)      // %57
                : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );                     
            }

            if (remain >= 8)
            {
                remain -= 8;

            asm volatile(
                "prfm   pldl1keep, [%9, #128]        \n"
                "prfm   pldl1keep, [%10, #128]       \n"
                "prfm   pldl1keep, [%11, #128]       \n"
                "prfm   pldl1keep, [%12, #128]       \n"
                "ld1    {v8.8b}, [%9], #8            \n" // r0"
                "ld1    {v9.8b}, [%10], #8           \n" // r1"
                "ld1    {v10.8b}, [%11], #8          \n" // r2"
                "ld1    {v11.8b}, [%12], #8          \n" // r3"

                "dup    v24.8b, %50.b[0]             \n" // k00
                "dup    v25.8b, %50.b[1]             \n" // k01
                "dup    v26.8b, %50.b[2]             \n" // k02
                "dup    v27.8b, %50.b[3]             \n" // k03

                "smull  v28.8h, v8.8b, v24.8b        \n" // r0
                "prfm   pldl1keep, [%13, #128]       \n"
                "prfm   pldl1keep, [%14, #128]       \n"
                "prfm   pldl1keep, [%15, #128]       \n"

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "prfm   pldl1keep, [%16, #128]       \n"
                "ld1    {v12.8b}, [%13], #8          \n" // r4" 
                "ld1    {v13.8b}, [%14], #8          \n" // r5"

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "ld1    {v14.8b}, [%15], #8          \n" // r6"
                "ld1    {v15.8b}, [%16], #8          \n" // r7"                         
                "dup    v24.8b, %50.b[4]             \n" // k04

                "smlal  v28.8h, v11.8b, v27.8b       \n"
                "dup    v25.8b, %50.b[5]             \n" // k05
                "dup    v26.8b, %50.b[6]             \n" // k06
                "dup    v27.8b, %50.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n" // r4
                "prfm   pldl1keep, [%1, #128]        \n"
                "ld1    {v29.8h}, [%1]       	     \n" // sum0  
                "prfm   pldl1keep, [%17, #128]       \n"

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "prfm   pldl1keep, [%18, #128]       \n"
                "prfm   pldl1keep, [%19, #128]       \n"
                "prfm   pldl1keep, [%20, #128]       \n"
                "ld1    {v16.8b}, [%17], #8          \n" // r8" 

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "ld1    {v17.8b}, [%18], #8          \n" // r9"
                "ld1    {v18.8b}, [%19], #8          \n" // r10"
                "ld1    {v19.8b}, [%20], #8          \n" // r11"

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v24.8b, %50.b[8]             \n" // k08
                "dup    v25.8b, %50.b[9]             \n" // k09
                "dup    v26.8b, %50.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n" // r8
                "dup    v27.8b, %50.b[11]            \n" // k11
                "prfm   pldl1keep, [%21, #128]       \n"
                "prfm   pldl1keep, [%22, #128]       \n"

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "prfm   pldl1keep, [%23, #128]       \n"
                "prfm   pldl1keep, [%24, #128]       \n"
                "ld1    {v20.8b}, [%21], #8          \n" // r12"

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "ld1    {v21.8b}, [%22], #8          \n" // r13"
                "ld1    {v22.8b}, [%23], #8          \n" // r14"
                "ld1    {v23.8b}, [%24], #8          \n" // r15" 

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v24.8b, %50.b[12]            \n" // k12
                "dup    v25.8b, %50.b[13]            \n" // k13
                "dup    v26.8b, %50.b[14]            \n" // k14

                "smlal  v28.8h, v20.8b, v24.8b       \n" // r12
                "dup    v27.8b, %50.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %51.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %51.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "dup    v26.8b, %51.b[2]             \n" // k02

                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                
                "dup    v27.8b, %51.b[3]             \n" // k03

                "st1    {v29.8h}, [%1], #16  		 \n" // sum0
                //########################################### 
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %51.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %51.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %51.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %51.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%2, #128]        \n"
                "ld1    {v29.8h}, [%2]       	     \n" // sum1

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %51.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %51.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %51.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %51.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %51.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %51.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %51.b[14]            \n" // k14

                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %51.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %52.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %52.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
            
                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                

                "dup    v26.8b, %52.b[2]             \n" // k02
                "dup    v27.8b, %52.b[3]             \n" // k03  

                "st1    {v29.8h}, [%2], #16  		 \n"
                //########################################### // sum1

                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %52.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %52.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %52.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %52.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%3, #128]        \n"
                "ld1    {v29.8h}, [%3]       		 \n" // sum2 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %52.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %52.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %52.b[10]            \n" // k10

                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %52.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %52.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %52.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %52.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %52.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %53.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %53.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
            
                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "dup    v26.8b, %53.b[2]             \n" // k02

                "dup    v27.8b, %53.b[3]             \n" // k03

                "st1    {v29.8h}, [%3], #16  		 \n"
                //########################################### //sum 2

                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %53.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %53.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %53.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %53.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%4, #128]        \n"
                "ld1    {v29.8h}, [%4]       		 \n" // sum3 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %53.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %53.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %53.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %53.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %53.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %53.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %53.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %53.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %54.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %54.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
            
                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "dup    v26.8b, %54.b[2]             \n" // k02

                "dup    v27.8b, %54.b[3]             \n" // k03

                "st1    {v29.8h}, [%4], #16  \n"
                //########################################### // sum3
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %54.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %54.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %54.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %54.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%5, #128]        \n"
                "ld1    {v29.8h}, [%5]       		 \n" // sum4

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %54.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %54.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"    
                "dup    v26.8b, %54.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %54.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %54.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %54.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %54.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %54.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %55.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %55.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"
                "dup    v26.8b, %55.b[2]             \n" // k02 
            
                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "dup    v27.8b, %55.b[3]             \n" // k03

                "st1    {v29.8h}, [%5], #16  \n"
                //########################################### // sum4
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %55.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %55.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %55.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %55.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%6, #128]        \n"
                "ld1    {v29.8h}, [%6]       		 \n" // sum5  

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %55.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %55.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %55.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %55.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %55.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %55.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %55.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %55.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %56.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %56.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
            
                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "dup    v26.8b, %56.b[2]             \n" // k02

                "dup    v27.8b, %56.b[3]             \n" // k03

                "st1    {v29.8h}, [%6], #16  		 \n"
                //########################################### // sum5
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %56.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %56.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %56.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %56.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%7, #128]        \n"
                "ld1    {v29.8h}, [%7]       		 \n" // sum6 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %56.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %56.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"        
                "dup    v26.8b, %56.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %56.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %56.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %56.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %56.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %56.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %57.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %57.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "dup    v26.8b, %57.b[2]             \n" // k02

                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                
                "dup    v27.8b, %57.b[3]             \n" // k03

                "st1    {v29.8h}, [%7], #16  		 \n"
                //########################################### // sum6
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %57.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %57.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %57.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %57.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%8, #128]        \n"
                "ld1    {v29.8h}, [%8]       		 \n" // sum7 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %57.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %57.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %57.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %57.b[11]            \n" // k11
                
                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %57.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %57.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %57.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %57.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "prfm   pldl1keep, [%9, #128]        \n"
                "prfm   pldl1keep, [%10, #128]       \n"
                "ld1    {v8.8b}, [%9], #8            \n" // r0"

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "ld1    {v9.8b}, [%10], #8           \n" // r1"
                "prfm   pldl1keep, [%11, #128]       \n"
                "prfm   pldl1keep, [%12, #128]       \n"

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "ld1    {v10.8b}, [%11], #8          \n" // r2"
                "ld1    {v11.8b}, [%12], #8          \n" // r3"
                "dup    v24.8b, %50.b[0]             \n" // k00                     

                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "dup    v25.8b, %50.b[1]             \n" // k01

                "dup    v26.8b, %50.b[2]             \n" // k02
                "dup    v27.8b, %50.b[3]             \n" // k03                

                "st1    {v29.8h}, [%8], #16  		 \n"   
                //########################################### // sum7
                : "=r"(nn),     // %0
                  "=r"(outptr0),// %1
                  "=r"(outptr1),// %2
                  "=r"(outptr2),// %3
                  "=r"(outptr3),// %4
                  "=r"(outptr4),// %5
                  "=r"(outptr5),// %6
                  "=r"(outptr6),// %7
                  "=r"(outptr7),// %8
                  "=r"(r0),     // %9
                  "=r"(r1),     // %10
                  "=r"(r2),     // %11
                  "=r"(r3),     // %12
                  "=r"(r4),     // %13
                  "=r"(r5),     // %14
                  "=r"(r6),     // %15
                  "=r"(r7),     // %16
                  "=r"(r8),     // %17
                  "=r"(r9),     // %18
                  "=r"(r10),    // %19
                  "=r"(r11),    // %20
                  "=r"(r12),    // %21
                  "=r"(r13),    // %22
                  "=r"(r14),    // %23
                  "=r"(r15)     // %24
                : "0"(nn),
                  "1"(outptr0),
                  "2"(outptr1),
                  "3"(outptr2),
                  "4"(outptr3),
                  "5"(outptr4),
                  "6"(outptr5),
                  "7"(outptr6),
                  "8"(outptr7),
                  "9"(r0),
                  "10"(r1),
                  "11"(r2),
                  "12"(r3),
                  "13"(r4),
                  "14"(r5),
                  "15"(r6),
                  "16"(r7),
                  "17"(r8),
                  "18"(r9),
                  "19"(r10),
                  "20"(r11),
                  "21"(r12),
                  "22"(r13),
                  "23"(r14),
                  "24"(r15),
                  "w"(_k0),     // %50
                  "w"(_k1),     // %51
                  "w"(_k2),     // %52
                  "w"(_k3),     // %53
                  "w"(_k4),     // %54
                  "w"(_k5),     // %55
                  "w"(_k6),     // %56
                  "w"(_k7)      // %57
                : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            );                             
            }

            if (remain >= 4)
            {
                remain -= 4;

            asm volatile(
                "prfm   pldl1keep, [%9, #128]        \n"
                "prfm   pldl1keep, [%10, #128]       \n"
                "prfm   pldl1keep, [%11, #128]       \n"
                "prfm   pldl1keep, [%12, #128]       \n"
                "ld1    {v8.8b}, [%9], #8            \n" // r0"
                "ld1    {v9.8b}, [%10], #8           \n" // r1"
                "ld1    {v10.8b}, [%11], #8          \n" // r2"
                "ld1    {v11.8b}, [%12], #8          \n" // r3"

                "dup    v24.8b, %50.b[0]             \n" // k00
                "dup    v25.8b, %50.b[1]             \n" // k01
                "dup    v26.8b, %50.b[2]             \n" // k02
                "dup    v27.8b, %50.b[3]             \n" // k03

                "smull  v28.8h, v8.8b, v24.8b        \n" // r0
                "prfm   pldl1keep, [%13, #128]       \n"
                "prfm   pldl1keep, [%14, #128]       \n"
                "prfm   pldl1keep, [%15, #128]       \n"

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "prfm   pldl1keep, [%16, #128]       \n"
                "ld1    {v12.8b}, [%13], #8          \n" // r4" 
                "ld1    {v13.8b}, [%14], #8          \n" // r5"

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "ld1    {v14.8b}, [%15], #8          \n" // r6"
                "ld1    {v15.8b}, [%16], #8          \n" // r7"                         
                "dup    v24.8b, %50.b[4]             \n" // k04

                "smlal  v28.8h, v11.8b, v27.8b       \n"
                "dup    v25.8b, %50.b[5]             \n" // k05
                "dup    v26.8b, %50.b[6]             \n" // k06
                "dup    v27.8b, %50.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n" // r4
                "prfm   pldl1keep, [%1, #128]        \n"
                "ld1    {v29.8h}, [%1]               \n" // sum0  
                "prfm   pldl1keep, [%17, #128]       \n"

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "prfm   pldl1keep, [%18, #128]       \n"
                "prfm   pldl1keep, [%19, #128]       \n"
                "prfm   pldl1keep, [%20, #128]       \n"
                "ld1    {v16.8b}, [%17], #8          \n" // r8" 

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "ld1    {v17.8b}, [%18], #8          \n" // r9"
                "ld1    {v18.8b}, [%19], #8          \n" // r10"
                "ld1    {v19.8b}, [%20], #8          \n" // r11"

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v24.8b, %50.b[8]             \n" // k08
                "dup    v25.8b, %50.b[9]             \n" // k09
                "dup    v26.8b, %50.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n" // r8
                "dup    v27.8b, %50.b[11]            \n" // k11
                "prfm   pldl1keep, [%21, #128]       \n"
                "prfm   pldl1keep, [%22, #128]       \n"

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "prfm   pldl1keep, [%23, #128]       \n"
                "prfm   pldl1keep, [%24, #128]       \n"
                "ld1    {v20.8b}, [%21], #8          \n" // r12"

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "ld1    {v21.8b}, [%22], #8          \n" // r13"
                "ld1    {v22.8b}, [%23], #8          \n" // r14"
                "ld1    {v23.8b}, [%24], #8          \n" // r15" 

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v24.8b, %50.b[12]            \n" // k12
                "dup    v25.8b, %50.b[13]            \n" // k13
                "dup    v26.8b, %50.b[14]            \n" // k14

                "smlal  v28.8h, v20.8b, v24.8b       \n" // r12
                "dup    v27.8b, %50.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %51.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %51.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "dup    v26.8b, %51.b[2]             \n" // k02

                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "dup    v27.8b, %51.b[3]             \n" // k03

                "st1    {v29.4h}, [%1], #8           \n" // sum0
                //########################################### 
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %51.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %51.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %51.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %51.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%2, #128]        \n"
                "ld1    {v29.8h}, [%2]               \n" // sum1

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %51.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %51.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %51.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %51.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %51.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %51.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %51.b[14]            \n" // k14

                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %51.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %52.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %52.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "dup    v26.8b, %52.b[2]             \n" // k02

                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "dup    v27.8b, %52.b[3]             \n" // k03  

                "st1    {v29.4h}, [%2], #8           \n"
                //########################################### // sum1

                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %52.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %52.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %52.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %52.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%3, #128]        \n"
                "ld1    {v29.8h}, [%3]               \n" // sum2 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %52.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %52.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %52.b[10]            \n" // k10

                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %52.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %52.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %52.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %52.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %52.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %53.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %53.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "dup    v26.8b, %53.b[2]             \n" // k02

                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "dup    v27.8b, %53.b[3]             \n" // k03

                "st1    {v29.4h}, [%3], #8           \n"
                //########################################### //sum 2

                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %53.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %53.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %53.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %53.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%4, #128]        \n"
                "ld1    {v29.8h}, [%4]               \n" // sum3 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %53.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %53.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %53.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %53.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %53.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %53.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %53.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %53.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %54.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %54.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "dup    v26.8b, %54.b[2]             \n" // k02

                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "dup    v27.8b, %54.b[3]             \n" // k03

                "st1    {v29.4h}, [%4], #8           \n"
                //########################################### // sum3
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %54.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %54.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %54.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %54.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%5, #128]        \n"
                "ld1    {v29.8h}, [%5]               \n" // sum4

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %54.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %54.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"    
                "dup    v26.8b, %54.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %54.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %54.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %54.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %54.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %54.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %55.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %55.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"
                "dup    v26.8b, %55.b[2]             \n" // k02 
            
                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "dup    v27.8b, %55.b[3]             \n" // k03
                
                "st1    {v29.4h}, [%5], #8           \n"
                //########################################### // sum4
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %55.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %55.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %55.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %55.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%6, #128]        \n"
                "ld1    {v29.8h}, [%6]               \n" // sum5  

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %55.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %55.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %55.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %55.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %55.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %55.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %55.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %55.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %56.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %56.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "dup    v26.8b, %56.b[2]             \n" // k02

                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "dup    v27.8b, %56.b[3]             \n" // k03

                "st1    {v29.4h}, [%6], #8           \n"
                //########################################### // sum5
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %56.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %56.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %56.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %56.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%7, #128]        \n"
                "ld1    {v29.8h}, [%7]               \n" // sum6 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %56.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %56.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"        
                "dup    v26.8b, %56.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %56.b[11]            \n" // k11

                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %56.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %56.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %56.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %56.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "dup    v24.8b, %57.b[0]             \n" // k00

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "dup    v25.8b, %57.b[1]             \n" // k01

                "smlal  v28.8h, v23.8b, v27.8b       \n"                
                "dup    v26.8b, %57.b[2]             \n" // k02

                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                
                "dup    v27.8b, %57.b[3]             \n" // k03

                "st1    {v29.4h}, [%7], #8           \n"
                //########################################### // sum6
                "smull  v28.8h, v8.8b, v24.8b        \n"
                "dup    v24.8b, %57.b[4]             \n" // k04

                "smlal  v28.8h, v9.8b, v25.8b        \n"
                "dup    v25.8b, %57.b[5]             \n" // k05

                "smlal  v28.8h, v10.8b, v26.8b       \n"
                "dup    v26.8b, %57.b[6]             \n" // k06

                "smlal  v28.8h, v11.8b, v27.8b       \n"                
                "dup    v27.8b, %57.b[7]             \n" // k07

                "smlal  v28.8h, v12.8b, v24.8b       \n"
                "prfm   pldl1keep, [%8, #128]        \n"
                "ld1    {v29.8h}, [%8]               \n" // sum7 

                "smlal  v28.8h, v13.8b, v25.8b       \n"
                "dup    v24.8b, %57.b[8]             \n" // k08

                "smlal  v28.8h, v14.8b, v26.8b       \n"
                "dup    v25.8b, %57.b[9]             \n" // k09

                "smlal  v28.8h, v15.8b, v27.8b       \n"
                "dup    v26.8b, %57.b[10]            \n" // k10
                
                "smlal  v28.8h, v16.8b, v24.8b       \n"
                "dup    v27.8b, %57.b[11]            \n" // k11
                
                "smlal  v28.8h, v17.8b, v25.8b       \n"
                "dup    v24.8b, %57.b[12]            \n" // k12

                "smlal  v28.8h, v18.8b, v26.8b       \n"
                "dup    v25.8b, %57.b[13]            \n" // k13

                "smlal  v28.8h, v19.8b, v27.8b       \n"
                "dup    v26.8b, %57.b[14]            \n" // k14
                
                "smlal  v28.8h, v20.8b, v24.8b       \n"
                "dup    v27.8b, %57.b[15]            \n" // k15

                "smlal  v28.8h, v21.8b, v25.8b       \n"
                "sub    %9, %9, #4                   \n"

                "smlal  v28.8h, v22.8b, v26.8b       \n"
                "sub    %10, %10, #4                 \n"
                "sub    %11, %11, #4                 \n"
                "sub    %12, %12, #4                 \n"

                "smlal  v28.8h, v23.8b, v27.8b       \n"    
                "sub    %13, %13, #4                 \n"
                "sub    %14, %14, #4                 \n"
                "sub    %15, %15, #4                 \n"
                "sub    %16, %16, #4                 \n"

                "sqadd  v29.8h, v29.8h, v28.8h       \n"
                "sub    %17, %17, #4                 \n"
                "sub    %18, %18, #4                 \n"
                "sub    %19, %19, #4                 \n"
                "sub    %20, %20, #4                 \n"

                "st1    {v29.4h}, [%8], #8           \n"
                //########################################### // sum7
                "sub    %21, %21, #4                 \n"
                "sub    %22, %22, #4                 \n"
                "sub    %23, %23, #4                 \n"
                "sub    %24, %24, #4                 \n" 
                : "=r"(nn),     // %0
                  "=r"(outptr0),// %1
                  "=r"(outptr1),// %2
                  "=r"(outptr2),// %3
                  "=r"(outptr3),// %4
                  "=r"(outptr4),// %5
                  "=r"(outptr5),// %6
                  "=r"(outptr6),// %7
                  "=r"(outptr7),// %8
                  "=r"(r0),     // %9
                  "=r"(r1),     // %10
                  "=r"(r2),     // %11
                  "=r"(r3),     // %12
                  "=r"(r4),     // %13
                  "=r"(r5),     // %14
                  "=r"(r6),     // %15
                  "=r"(r7),     // %16
                  "=r"(r8),     // %17
                  "=r"(r9),     // %18
                  "=r"(r10),     // %19
                  "=r"(r11),     // %20
                  "=r"(r12),     // %21
                  "=r"(r13),     // %22
                  "=r"(r14),     // %23
                  "=r"(r15)      // %24
                : "0"(nn),
                  "1"(outptr0),
                  "2"(outptr1),
                  "3"(outptr2),
                  "4"(outptr3),
                  "5"(outptr4),
                  "6"(outptr5),
                  "7"(outptr6),
                  "8"(outptr7),
                  "9"(r0),
                  "10"(r1),
                  "11"(r2),
                  "12"(r3),
                  "13"(r4),
                  "14"(r5),
                  "15"(r6),
                  "16"(r7),
                  "17"(r8),
                  "18"(r9),
                  "19"(r10),
                  "20"(r11),
                  "21"(r12),
                  "22"(r13),
                  "23"(r14),
                  "24"(r15),
                  "w"(_k0),     // %50
                  "w"(_k1),     // %51
                  "w"(_k2),     // %52
                  "w"(_k3),     // %53
                  "w"(_k4),     // %54
                  "w"(_k5),     // %55
                  "w"(_k6),     // %56
                  "w"(_k7)      // %57
                : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
            ); 
            }

            for (; remain>0; remain--)
            {
                // TODO neon optimize
                short sum0 = (short)*r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3] + *r4 * kernel0[4] + *r5 * kernel0[5] + *r6 * kernel0[6] + *r7 * kernel0[7] + *r8 * kernel0[8] + *r9 * kernel0[9] + *r10 * kernel0[10] + *r11 * kernel0[11] + *r12 * kernel0[12] + *r13 * kernel0[13] + *r14 * kernel0[14] + *r15 * kernel0[15];
                short sum1 = (short)*r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3] + *r4 * kernel1[4] + *r5 * kernel1[5] + *r6 * kernel1[6] + *r7 * kernel1[7] + *r8 * kernel1[8] + *r9 * kernel1[9] + *r10 * kernel1[10] + *r11 * kernel1[11] + *r12 * kernel1[12] + *r13 * kernel1[13] + *r14 * kernel1[14] + *r15 * kernel1[15];
                short sum2 = (short)*r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3] + *r4 * kernel2[4] + *r5 * kernel2[5] + *r6 * kernel2[6] + *r7 * kernel2[7] + *r8 * kernel2[8] + *r9 * kernel2[9] + *r10 * kernel2[10] + *r11 * kernel2[11] + *r12 * kernel2[12] + *r13 * kernel2[13] + *r14 * kernel2[14] + *r15 * kernel2[15];
                short sum3 = (short)*r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3] + *r4 * kernel3[4] + *r5 * kernel3[5] + *r6 * kernel3[6] + *r7 * kernel3[7] + *r8 * kernel3[8] + *r9 * kernel3[9] + *r10 * kernel3[10] + *r11 * kernel3[11] + *r12 * kernel3[12] + *r13 * kernel3[13] + *r14 * kernel3[14] + *r15 * kernel3[15];
                short sum4 = (short)*r0 * kernel4[0] + *r1 * kernel4[1] + *r2 * kernel4[2] + *r3 * kernel4[3] + *r4 * kernel4[4] + *r5 * kernel4[5] + *r6 * kernel4[6] + *r7 * kernel4[7] + *r8 * kernel4[8] + *r9 * kernel4[9] + *r10 * kernel4[10] + *r11 * kernel4[11] + *r12 * kernel4[12] + *r13 * kernel4[13] + *r14 * kernel4[14] + *r15 * kernel4[15];
                short sum5 = (short)*r0 * kernel5[0] + *r1 * kernel5[1] + *r2 * kernel5[2] + *r3 * kernel5[3] + *r4 * kernel5[4] + *r5 * kernel5[5] + *r6 * kernel5[6] + *r7 * kernel5[7] + *r8 * kernel5[8] + *r9 * kernel5[9] + *r10 * kernel5[10] + *r11 * kernel5[11] + *r12 * kernel5[12] + *r13 * kernel5[13] + *r14 * kernel5[14] + *r15 * kernel5[15];
                short sum6 = (short)*r0 * kernel6[0] + *r1 * kernel6[1] + *r2 * kernel6[2] + *r3 * kernel6[3] + *r4 * kernel6[4] + *r5 * kernel6[5] + *r6 * kernel6[6] + *r7 * kernel6[7] + *r8 * kernel6[8] + *r9 * kernel6[9] + *r10 * kernel6[10] + *r11 * kernel6[11] + *r12 * kernel6[12] + *r13 * kernel6[13] + *r14 * kernel6[14] + *r15 * kernel6[15];
                short sum7 = (short)*r0 * kernel7[0] + *r1 * kernel7[1] + *r2 * kernel7[2] + *r3 * kernel7[3] + *r4 * kernel7[4] + *r5 * kernel7[5] + *r6 * kernel7[6] + *r7 * kernel7[7] + *r8 * kernel7[8] + *r9 * kernel7[9] + *r10 * kernel7[10] + *r11 * kernel7[11] + *r12 * kernel7[12] + *r13 * kernel7[13] + *r14 * kernel7[14] + *r15 * kernel7[15];

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;
                *outptr4 += sum4;
                *outptr5 += sum5;
                *outptr6 += sum6;
                *outptr7 += sum7;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;
                r8++;
                r9++;
                r10++;
                r11++;
                r12++;
                r13++;
                r14++;
                r15++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
                outptr4++;
                outptr5++;
                outptr6++;
                outptr7++;          
            }
        }
#else
        for (; q+7<inch; q+=8)
        {
            short* outptr0 = out0;
            short* outptr1 = out1;
            short* outptr2 = out2;
            short* outptr3 = out3;
            short* outptr4 = out4;
            short* outptr5 = out5;
            short* outptr6 = out6;
            short* outptr7 = out7;

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;
            const signed char* kernel1 = (const signed char*)kernel + (p+1)*inch + q;
            const signed char* kernel2 = (const signed char*)kernel + (p+2)*inch + q;
            const signed char* kernel3 = (const signed char*)kernel + (p+3)*inch + q;
            const signed char* kernel4 = (const signed char*)kernel + (p+4)*inch + q;
            const signed char* kernel5 = (const signed char*)kernel + (p+5)*inch + q;
            const signed char* kernel6 = (const signed char*)kernel + (p+6)*inch + q;
            const signed char* kernel7 = (const signed char*)kernel + (p+7)*inch + q;

            const signed char* r0 = bottom_blob.channel(q);
            const signed char* r1 = bottom_blob.channel(q+1);
            const signed char* r2 = bottom_blob.channel(q+2);
            const signed char* r3 = bottom_blob.channel(q+3);
            const signed char* r4 = bottom_blob.channel(q+4);
            const signed char* r5 = bottom_blob.channel(q+5);
            const signed char* r6 = bottom_blob.channel(q+6);
            const signed char* r7 = bottom_blob.channel(q+7);

            int size = outw * outh;

            int nn = size >> 4;
            int remain = size & 15;

            asm volatile(
                "ld1    {v0.16b}, [%0]    \n"
                "ld1    {v1.16b}, [%1]    \n"
                "ld1    {v2.16b}, [%2]    \n"
                "ld1    {v3.16b}, [%3]    \n"
                "ld1    {v4.16b}, [%4]    \n"
                "ld1    {v5.16b}, [%5]    \n"
                "ld1    {v6.16b}, [%6]    \n"
                "ld1    {v7.16b}, [%7]    \n"
                : 
                : "r"(kernel0),
                  "r"(kernel1),
                  "r"(kernel2),
                  "r"(kernel3),
                  "r"(kernel4),
                  "r"(kernel5),
                  "r"(kernel6),
                  "r"(kernel7)
                : "cc", "memory"
            );

	        if (nn > 0)
            {
            asm volatile(
                "prfm   pldl1keep, [%18, #128]       \n"
                "prfm   pldl1keep, [%19, #128]       \n"
                "prfm   pldl1keep, [%20, #128]       \n"
                "prfm   pldl1keep, [%21, #128]       \n"
                "prfm   pldl1keep, [%22, #128]       \n"
                "prfm   pldl1keep, [%23, #128]       \n"
                "prfm   pldl1keep, [%24, #128]       \n"
                "prfm   pldl1keep, [%25, #128]       \n"
                "ld1    {v8.16b}, [%18], #16         \n" // r0"
                "ld1    {v9.16b}, [%19], #16         \n" // r1"
                "ld1    {v10.16b}, [%20], #16        \n" // r2"
                "ld1    {v11.16b}, [%21], #16        \n" // r3"
                "ld1    {v12.16b}, [%22], #16        \n" // r4"
                "ld1    {v13.16b}, [%23], #16        \n" // r5"
                "ld1    {v14.16b}, [%24], #16        \n" // r6"
                "ld1    {v15.16b}, [%25], #16        \n" // r7"
                
                "0:                                  \n"

                "dup    v16.16b, v0.b[0]           \n" // k00
                "dup    v17.16b, v0.b[1]           \n" // k01
                "dup    v18.16b, v0.b[2]           \n" // k02
                "dup    v19.16b, v0.b[3]           \n" // k03
                "dup    v20.16b, v0.b[4]           \n" // k04
                "dup    v21.16b, v0.b[5]           \n" // k05
                "dup    v22.16b, v0.b[6]           \n" // k06
                "dup    v23.16b, v0.b[7]           \n" // k07				

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smull2 v25.8h, v8.16b, v16.16b      \n" 
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal2  v25.8h, v9.16b, v17.16b     \n" 
                "dup    v16.16b, v1.b[0]           \n" // k00
                
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal2  v25.8h, v10.16b, v18.16b    \n"
                "dup    v17.16b, v1.b[1]           \n" // k01
                
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal2  v25.8h, v11.16b, v19.16b    \n"
                "dup    v18.16b, v1.b[2]           \n" // k02

                "prfm   pldl1keep, [%1, #128]        \n"
                "ld1    {v26.8h, v27.8h}, [%1]       \n" // sum0  
                                    
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal2  v25.8h, v12.16b, v20.16b    \n"
                "dup    v19.16b, v1.b[3]           \n" // k03
                
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal2  v25.8h, v13.16b, v21.16b    \n"
                "dup    v20.16b, v1.b[4]           \n" // k04
                
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal2  v25.8h, v14.16b, v22.16b    \n"
                "dup    v21.16b, v1.b[5]           \n" // k05
                
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                "smlal2  v25.8h, v15.16b, v23.16b    \n"      
                "dup    v22.16b, v1.b[6]           \n" // k06

                "sqadd  v26.8h, v26.8h, v24.8h       \n"
                "sqadd  v27.8h, v27.8h, v25.8h       \n"
                "dup    v23.16b, v1.b[7]           \n" // k07	

                "st1    {v26.8h, v27.8h}, [%1], #32  \n" // sum0n
                //###########################################
                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smull2 v25.8h, v8.16b, v16.16b      \n" 
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal2  v25.8h, v9.16b, v17.16b     \n"
                "dup    v16.16b, v2.b[0]           \n" // k00
                
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal2  v25.8h, v10.16b, v18.16b    \n"
                "dup    v17.16b, v2.b[1]           \n" // k01
                
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal2  v25.8h, v11.16b, v19.16b    \n"
                "dup    v18.16b, v2.b[2]           \n" // k02

                "prfm   pldl1keep, [%2, #128]        \n"
                "ld1    {v26.8h, v27.8h}, [%2]       \n" // sum1

                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal2  v25.8h, v12.16b, v20.16b    \n"
                "dup    v19.16b, v2.b[3]           \n" // k03
                
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal2  v25.8h, v13.16b, v21.16b    \n"
                "dup    v20.16b, v2.b[4]           \n" // k04
                
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal2  v25.8h, v14.16b, v22.16b    \n"  
                "dup    v21.16b, v2.b[5]           \n" // k05
                
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                "smlal2  v25.8h, v15.16b, v23.16b    \n"    
                "dup    v22.16b, v2.b[6]           \n" // k06  
                
                "sqadd  v26.8h, v26.8h, v24.8h       \n"
                "sqadd  v27.8h, v27.8h, v25.8h       \n"
                "dup    v23.16b, v2.b[7]           \n" // k07

                "st1    {v26.8h, v27.8h}, [%2], #32  \n" // sum1n
                //###########################################

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smull2 v25.8h, v8.16b, v16.16b      \n" 
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal2  v25.8h, v9.16b, v17.16b     \n" 
                "dup    v16.16b, v3.b[0]           \n" // k00
                
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal2  v25.8h, v10.16b, v18.16b    \n"
                "dup    v17.16b, v3.b[1]           \n" // k01
                
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal2  v25.8h, v11.16b, v19.16b    \n"
                "dup    v18.16b, v3.b[2]           \n" // k02					

                "prfm   pldl1keep, [%3, #128]        \n"
                "ld1    {v26.8h, v27.8h}, [%3]       \n" // sum2

                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal2  v25.8h, v12.16b, v20.16b    \n"
                "dup    v19.16b, v3.b[3]           \n" // k03
                
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal2  v25.8h, v13.16b, v21.16b    \n"
                "dup    v20.16b, v3.b[4]           \n" // k04
                
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal2  v25.8h, v14.16b, v22.16b    \n"  
                "dup    v21.16b, v3.b[5]           \n" // k05
                
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                "smlal2  v25.8h, v15.16b, v23.16b    \n"    
                "dup    v22.16b, v3.b[6]           \n" // k06  
                
                "sqadd  v26.8h, v26.8h, v24.8h       \n"
                "sqadd  v27.8h, v27.8h, v25.8h       \n"
                "dup    v23.16b, v3.b[7]           \n" // k07

                "st1    {v26.8h, v27.8h}, [%3], #32  \n" // sum2n  					
                //##########################################
                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smull2 v25.8h, v8.16b, v16.16b      \n" 
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal2  v25.8h, v9.16b, v17.16b     \n" 
                "dup    v16.16b, v4.b[0]           \n" // k00
                
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal2  v25.8h, v10.16b, v18.16b    \n"
                "dup    v17.16b, v4.b[1]            \n" // k01
                
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal2  v25.8h, v11.16b, v19.16b    \n"
                "dup    v18.16b, v4.b[2]           \n" // k02					

                "prfm   pldl1keep, [%4, #128]        \n"
                "ld1    {v26.8h, v27.8h}, [%4]       \n" // sum3

                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal2  v25.8h, v12.16b, v20.16b    \n"
                "dup    v19.16b, v4.b[3]           \n" // k03
                
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal2  v25.8h, v13.16b, v21.16b    \n"
                "dup    v20.16b, v4.b[4]            \n" // k04
                
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal2  v25.8h, v14.16b, v22.16b    \n" 
                "dup    v21.16b, v4.b[5]           \n" // k05
                
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                "smlal2  v25.8h, v15.16b, v23.16b    \n"      
                "dup    v22.16b, v4.b[6]           \n" // k06
                
                "sqadd  v26.8h, v26.8h, v24.8h       \n"
                "sqadd  v27.8h, v27.8h, v25.8h       \n"
                "dup    v23.16b, v4.b[7]           \n" // k07	

                "st1    {v26.8h, v27.8h}, [%4], #32  \n" // sum3n
                //##########################################	
                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smull2 v25.8h, v8.16b, v16.16b      \n" 
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal2  v25.8h, v9.16b, v17.16b     \n" 
                "dup    v16.16b, v5.b[0]           \n" // k00
                
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal2  v25.8h, v10.16b, v18.16b    \n"
                "dup    v17.16b, v5.b[1]           \n" // k01
                
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal2  v25.8h, v11.16b, v19.16b    \n"
                "dup    v18.16b, v5.b[2]           \n" // k02

                "prfm   pldl1keep, [%5, #128]        \n"
                "ld1    {v26.8h, v27.8h}, [%5]       \n" // sum4

                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal2  v25.8h, v12.16b, v20.16b    \n"
                "dup    v19.16b, v5.b[3]           \n" // k03
                
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal2  v25.8h, v13.16b, v21.16b    \n"
                "dup    v20.16b, v5.b[4]           \n" // k04
                
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal2  v25.8h, v14.16b, v22.16b    \n" 
                "dup    v21.16b, v5.b[5]           \n" // k05
                
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                "smlal2  v25.8h, v15.16b, v23.16b    \n"      
                "dup    v22.16b, v5.b[6]           \n" // k06
                
                "sqadd  v26.8h, v26.8h, v24.8h       \n"
                "sqadd  v27.8h, v27.8h, v25.8h       \n"
                "dup    v23.16b, v5.b[7]           \n" // k07

                "st1    {v26.8h, v27.8h}, [%5], #32  \n" // sum4n
                //##########################################	
                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smull2 v25.8h, v8.16b, v16.16b      \n" 
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal2  v25.8h, v9.16b, v17.16b     \n" 
                "dup    v16.16b, v6.b[0]           \n" // k00
                
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal2  v25.8h, v10.16b, v18.16b    \n"
                "dup    v17.16b, v6.b[1]           \n" // k01
                
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal2  v25.8h, v11.16b, v19.16b    \n"
                "dup    v18.16b, v6.b[2]           \n" // k02

                "prfm   pldl1keep, [%6, #128]        \n"
                "ld1    {v26.8h, v27.8h}, [%6]       \n" // sum5                
                
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal2  v25.8h, v12.16b, v20.16b    \n"
                "dup    v19.16b, v6.b[3]           \n" // k03
                
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal2  v25.8h, v13.16b, v21.16b    \n"
                "dup    v20.16b, v6.b[4]           \n" // k04
                
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal2  v25.8h, v14.16b, v22.16b    \n"  
                "dup    v21.16b, v6.b[5]           \n" // k05
                
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                "smlal2  v25.8h, v15.16b, v23.16b    \n"      
                "dup    v22.16b, v6.b[6]           \n" // k06
                
                "sqadd  v26.8h, v26.8h, v24.8h       \n"
                "sqadd  v27.8h, v27.8h, v25.8h       \n"
                "dup    v23.16b, v6.b[7]           \n" // k07

                "st1    {v26.8h, v27.8h}, [%6], #32  \n" // sum5n
                //##########################################
                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smull2 v25.8h, v8.16b, v16.16b      \n" 
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal2  v25.8h, v9.16b, v17.16b     \n" 
                "dup    v16.16b, v7.b[0]           \n" // k00
                
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal2  v25.8h, v10.16b, v18.16b    \n"
                "dup    v17.16b, v7.b[1]           \n" // k01
                
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal2  v25.8h, v11.16b, v19.16b    \n"
                "dup    v18.16b, v7.b[2]           \n" // k02					

                "prfm   pldl1keep, [%7, #128]        \n"
                "ld1    {v26.8h, v27.8h}, [%7]       \n" // sum6

                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal2  v25.8h, v12.16b, v20.16b    \n"
                "dup    v19.16b, v7.b[3]           \n" // k03
                
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal2  v25.8h, v13.16b, v21.16b    \n"
                "dup    v20.16b, v7.b[4]           \n" // k04
                
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal2  v25.8h, v14.16b, v22.16b    \n"
                "dup    v21.16b, v7.b[5]           \n" // k05
                
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                "smlal2  v25.8h, v15.16b, v23.16b    \n"   
                "dup    v22.16b, v7.b[6]           \n" // k06   
                
                "sqadd  v26.8h, v26.8h, v24.8h       \n"
                "sqadd  v27.8h, v27.8h, v25.8h       \n"
                "dup    v23.16b, v7.b[7]           \n" // k07

                "st1    {v26.8h, v27.8h}, [%7], #32  \n" // sum6n
                //##########################################		
                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smull2 v25.8h, v8.16b, v16.16b      \n" 
                "prfm   pldl1keep, [%18, #128]       \n"
                "prfm   pldl1keep, [%19, #128]       \n"
                
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal2  v25.8h, v9.16b, v17.16b     \n" 
                "prfm   pldl1keep, [%20, #128]       \n"
                "prfm   pldl1keep, [%21, #128]       \n"
                
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal2  v25.8h, v10.16b, v18.16b    \n"
                "prfm   pldl1keep, [%22, #128]       \n"
                "prfm   pldl1keep, [%23, #128]       \n"
                
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal2  v25.8h, v11.16b, v19.16b    \n"

                "prfm   pldl1keep, [%8, #128]        \n"
                "ld1    {v26.8h, v27.8h}, [%8]       \n" // sum7
                                    
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal2  v25.8h, v12.16b, v20.16b    \n"
                "prfm   pldl1keep, [%24, #128]       \n"
                "prfm   pldl1keep, [%25, #128]       \n"
                
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal2  v25.8h, v13.16b, v21.16b    \n"
                "ld1    {v8.16b}, [%18], #16         \n" // r0"
                "ld1    {v9.16b}, [%19], #16         \n" // r1"
                
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal2  v25.8h, v14.16b, v22.16b    \n"  
                "ld1    {v10.16b}, [%20], #16        \n" // r2"
                "ld1    {v11.16b}, [%21], #16        \n" // r3"
                
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                "smlal2  v25.8h, v15.16b, v23.16b    \n"   
                "ld1    {v12.16b}, [%22], #16        \n" // r4"
                "ld1    {v13.16b}, [%23], #16        \n" // r5"					
                
                "sqadd  v26.8h, v26.8h, v24.8h       \n"
                "sqadd  v27.8h, v27.8h, v25.8h       \n"	
                "ld1    {v14.16b}, [%24], #16        \n" // r6"
                "ld1    {v15.16b}, [%25], #16        \n" // r7"	

                "st1    {v26.8h, v27.8h}, [%8], #32  \n" // sum7n

                "subs   %w0, %w0, #1                 \n"
                "bne    0b                           \n"
                "sub    %18, %18, #16                \n"
                "sub    %19, %19, #16                \n"
                "sub    %20, %20, #16                \n"
                "sub    %21, %21, #16                \n"
                "sub    %22, %22, #16                \n"
                "sub    %23, %23, #16                \n"
                "sub    %24, %24, #16                \n"
                "sub    %25, %25, #16                \n"
                //##########################################					
                : "=r"(nn),     // %0
                  "=r"(outptr0),// %1
                  "=r"(outptr1),// %2
                  "=r"(outptr2),// %3
                  "=r"(outptr3),// %4
                  "=r"(outptr4),// %5
                  "=r"(outptr5),// %6
                  "=r"(outptr6),// %7
                  "=r"(outptr7) // %8
                : "0"(nn),      
                  "1"(outptr0),
                  "2"(outptr1),
                  "3"(outptr2),
                  "4"(outptr3),
                  "5"(outptr4),
                  "6"(outptr5),
                  "7"(outptr6),
                  "8"(outptr7),
                  "r"(r0),      // %18
                  "r"(r1),		// %19
                  "r"(r2),		// %20
                  "r"(r3),		// %21
                  "r"(r4),		// %22
                  "r"(r5),		// %23
                  "r"(r6),		// %24
                  "r"(r7)		// %25
                : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29"
            );
			}
#if 0
            if (remain == 8)
            {
                remain -= 8;

            asm volatile(
                "prfm   pldl1keep, [%18, #128]       \n"
                "prfm   pldl1keep, [%19, #128]       \n"
                "prfm   pldl1keep, [%20, #128]       \n"
                "prfm   pldl1keep, [%21, #128]       \n"
                "prfm   pldl1keep, [%22, #128]       \n"
                "prfm   pldl1keep, [%23, #128]       \n"
                "prfm   pldl1keep, [%24, #128]       \n"
                "prfm   pldl1keep, [%25, #128]       \n"				
                "ld1    {v8.8b}, [%18], #8           \n" // r0"
                "ld1    {v9.8b}, [%19], #8           \n" // r1"
                "ld1    {v10.8b}, [%20], #8          \n" // r2"
                "ld1    {v11.8b}, [%21], #8          \n" // r3"
                "ld1    {v12.8b}, [%22], #8          \n" // r4"   
                "ld1    {v13.8b}, [%23], #8          \n" // r5"	
                "ld1    {v14.8b}, [%24], #8          \n" // r6"
                "ld1    {v15.8b}, [%25], #8          \n" // r7" 

                "dup    v16.8b, v0.16b[0]            \n" // k00
                "dup    v17.8b, v0.16b[1]            \n" // k01
                "dup    v18.8b, v0.16b[2]            \n" // k02
                "dup    v19.8b, v0.16b[3]            \n" // k03
                "dup    v20.8b, v0.16b[4]            \n" // k04
                "dup    v21.8b, v0.16b[5]            \n" // k05
                "dup    v22.8b, v0.16b[6]            \n" // k06
                "dup    v23.8b, v0.16b[7]            \n" // k07				

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%1, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%1]       \n" // sum0  
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%1], #32  \n" 
                //###########################################
                "dup    v16.8b, v1.16b[0]            \n" // k00
                "dup    v17.8b, v1.16b[1]            \n" // k01
                "dup    v18.8b, v1.16b[2]            \n" // k02
                "dup    v19.8b, v1.16b[3]            \n" // k03
                "dup    v20.8b, v1.16b[4]            \n" // k04
                "dup    v21.8b, v1.16b[5]            \n" // k05
                "dup    v22.8b, v1.16b[6]            \n" // k06
                "dup    v23.8b, v1.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%2, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%2]       \n" // sum1
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%2], #32  \n" 
                //###########################################
                "dup    v16.8b, v2.16b[0]            \n" // k00
                "dup    v17.8b, v2.16b[1]            \n" // k01
                "dup    v18.8b, v2.16b[2]            \n" // k02
                "dup    v19.8b, v2.16b[3]            \n" // k03
                "dup    v20.8b, v2.16b[4]            \n" // k04
                "dup    v21.8b, v2.16b[5]            \n" // k05
                "dup    v22.8b, v2.16b[6]            \n" // k06
                "dup    v23.8b, v2.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%3, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%3]       \n" // sum2
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%3], #32  \n" 
                //##########################################
                "dup    v16.8b, v3.16b[0]            \n" // k00
                "dup    v17.8b, v3.16b[1]            \n" // k01
                "dup    v18.8b, v3.16b[2]            \n" // k02
                "dup    v19.8b, v3.16b[3]            \n" // k03
                "dup    v20.8b, v3.16b[4]            \n" // k04
                "dup    v21.8b, v3.16b[5]            \n" // k05
                "dup    v22.8b, v3.16b[6]            \n" // k06
                "dup    v23.8b, v3.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%4, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%4]       \n" // sum3
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%4], #32  \n" 
                //##########################################	
                "dup    v16.8b, v4.16b[0]            \n" // k00
                "dup    v17.8b, v4.16b[1]            \n" // k01
                "dup    v18.8b, v4.16b[2]            \n" // k02
                "dup    v19.8b, v4.16b[3]            \n" // k03
                "dup    v20.8b, v4.16b[4]            \n" // k04
                "dup    v21.8b, v4.16b[5]            \n" // k05
                "dup    v22.8b, v4.16b[6]            \n" // k06
                "dup    v23.8b, v4.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%5, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%5]       \n" // sum4
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%5], #32  \n" 
                //##########################################	
                "dup    v16.8b, v5.16b[0]            \n" // k00
                "dup    v17.8b, v5.16b[1]            \n" // k01
                "dup    v18.8b, v5.16b[2]            \n" // k02
                "dup    v19.8b, v5.16b[3]            \n" // k03
                "dup    v20.8b, v5.16b[4]            \n" // k04
                "dup    v21.8b, v5.16b[5]            \n" // k05
                "dup    v22.8b, v5.16b[6]            \n" // k06
                "dup    v23.8b, v5.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%6, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%6]       \n" // sum5
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%6], #32  \n" 
                //##########################################
                "dup    v16.8b, v6.16b[0]            \n" // k00
                "dup    v17.8b, v6.16b[1]            \n" // k01
                "dup    v18.8b, v6.16b[2]            \n" // k02
                "dup    v19.8b, v6.16b[3]            \n" // k03
                "dup    v20.8b, v6.16b[4]            \n" // k04
                "dup    v21.8b, v6.16b[5]            \n" // k05
                "dup    v22.8b, v6.16b[6]            \n" // k06
                "dup    v23.8b, v6.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%7, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%7]       \n" // sum6
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%7], #32  \n" 
                //##########################################		
                "dup    v16.8b, v7.16b[0]            \n" // k00
                "dup    v17.8b, v7.16b[1]            \n" // k01
                "dup    v18.8b, v7.16b[2]            \n" // k02
                "dup    v19.8b, v7.16b[3]            \n" // k03
                "dup    v20.8b, v7.16b[4]            \n" // k04
                "dup    v21.8b, v7.16b[5]            \n" // k05
                "dup    v22.8b, v7.16b[6]            \n" // k06
                "dup    v23.8b, v7.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%8, #128]        \n"
                "ld1    {v26.4s, v27.4s}, [%8]       \n" // sum7
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "saddw2 v27.4s, v27.4s, v24.8h       \n"
                "st1    {v26.4s, v27.4s}, [%8], #32  \n" 
                //##########################################					
                : "=r"(nn),     // %0
                  "=r"(outptr0),// %1
                  "=r"(outptr1),// %2
                  "=r"(outptr2),// %3
                  "=r"(outptr3),// %4
                  "=r"(outptr4),// %5
                  "=r"(outptr5),// %6
                  "=r"(outptr6),// %7
                  "=r"(outptr7) // %8
                : "0"(nn),      
                  "1"(outptr0),
                  "2"(outptr1),
                  "3"(outptr2),
                  "4"(outptr3),
                  "5"(outptr4),
                  "6"(outptr5),
                  "7"(outptr6),
                  "8"(outptr7),
                  "r"(r0),              // %18
                  "r"(r1),		// %19
                  "r"(r2),		// %20
                  "r"(r3),		// %21
                  "r"(r4),		// %22
                  "r"(r5),		// %23
                  "r"(r6),		// %24
                  "r"(r7)		// %25
                : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29"
            );
			}                               

            if (remain == 4)
            {
                remain -= 4;

            asm volatile(		
                "ld1    {v8.8b}, [%18], #8           \n" // r0"
                "ld1    {v9.8b}, [%19], #8           \n" // r1"
                "ld1    {v10.8b}, [%20], #8          \n" // r2"
                "ld1    {v11.8b}, [%21], #8          \n" // r3"
                "ld1    {v12.8b}, [%22], #8          \n" // r4"   
                "ld1    {v13.8b}, [%23], #8          \n" // r5"	
                "ld1    {v14.8b}, [%24], #8          \n" // r6"
                "ld1    {v15.8b}, [%25], #8          \n" // r7" 

                "dup    v16.8b, v0.16b[0]            \n" // k00
                "dup    v17.8b, v0.16b[1]            \n" // k01
                "dup    v18.8b, v0.16b[2]            \n" // k02
                "dup    v19.8b, v0.16b[3]            \n" // k03
                "dup    v20.8b, v0.16b[4]            \n" // k04
                "dup    v21.8b, v0.16b[5]            \n" // k05
                "dup    v22.8b, v0.16b[6]            \n" // k06
                "dup    v23.8b, v0.16b[7]            \n" // k07				

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%1, #128]        \n"
                "ld1    {v26.4s}, [%1]               \n" // sum0  
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "st1    {v26.4s}, [%1], #16  	     \n" 
                //###########################################
                "dup    v16.8b, v1.16b[0]            \n" // k00
                "dup    v17.8b, v1.16b[1]            \n" // k01
                "dup    v18.8b, v1.16b[2]            \n" // k02
                "dup    v19.8b, v1.16b[3]            \n" // k03
                "dup    v20.8b, v1.16b[4]            \n" // k04
                "dup    v21.8b, v1.16b[5]            \n" // k05
                "dup    v22.8b, v1.16b[6]            \n" // k06
                "dup    v23.8b, v1.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%2, #128]        \n"
                "ld1    {v26.4s}, [%2]               \n" // sum1
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "st1    {v26.4s}, [%2], #16  	     \n" 
                //###########################################
                "dup    v16.8b, v2.16b[0]            \n" // k00
                "dup    v17.8b, v2.16b[1]            \n" // k01
                "dup    v18.8b, v2.16b[2]            \n" // k02
                "dup    v19.8b, v2.16b[3]            \n" // k03
                "dup    v20.8b, v2.16b[4]            \n" // k04
                "dup    v21.8b, v2.16b[5]            \n" // k05
                "dup    v22.8b, v2.16b[6]            \n" // k06
                "dup    v23.8b, v2.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%3, #128]        \n"
                "ld1    {v26.4s}, [%3]       	     \n" // sum2
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "st1    {v26.4s}, [%3], #16  	     \n" 
                //##########################################
                "dup    v16.8b, v3.16b[0]            \n" // k00
                "dup    v17.8b, v3.16b[1]            \n" // k01
                "dup    v18.8b, v3.16b[2]            \n" // k02
                "dup    v19.8b, v3.16b[3]            \n" // k03
                "dup    v20.8b, v3.16b[4]            \n" // k04
                "dup    v21.8b, v3.16b[5]            \n" // k05
                "dup    v22.8b, v3.16b[6]            \n" // k06
                "dup    v23.8b, v3.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%4, #128]        \n"
                "ld1    {v26.4s}, [%4]       	     \n" // sum3
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "st1    {v26.4s}, [%4], #16  	     \n" 
                //##########################################	
                "dup    v16.8b, v4.16b[0]            \n" // k00
                "dup    v17.8b, v4.16b[1]            \n" // k01
                "dup    v18.8b, v4.16b[2]            \n" // k02
                "dup    v19.8b, v4.16b[3]            \n" // k03
                "dup    v20.8b, v4.16b[4]            \n" // k04
                "dup    v21.8b, v4.16b[5]            \n" // k05
                "dup    v22.8b, v4.16b[6]            \n" // k06
                "dup    v23.8b, v4.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%5, #128]        \n"
                "ld1    {v26.4s}, [%5]       	     \n" // sum4
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "st1    {v26.4s}, [%5], #16  	     \n" 
                //##########################################	
                "dup    v16.8b, v5.16b[0]            \n" // k00
                "dup    v17.8b, v5.16b[1]            \n" // k01
                "dup    v18.8b, v5.16b[2]            \n" // k02
                "dup    v19.8b, v5.16b[3]            \n" // k03
                "dup    v20.8b, v5.16b[4]            \n" // k04
                "dup    v21.8b, v5.16b[5]            \n" // k05
                "dup    v22.8b, v5.16b[6]            \n" // k06
                "dup    v23.8b, v5.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%6, #128]        \n"
                "ld1    {v26.4s}, [%6]       	     \n" // sum5
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "st1    {v26.4s}, [%6], #16  	     \n" 
                //##########################################
                "dup    v16.8b, v6.16b[0]            \n" // k00
                "dup    v17.8b, v6.16b[1]            \n" // k01
                "dup    v18.8b, v6.16b[2]            \n" // k02
                "dup    v19.8b, v6.16b[3]            \n" // k03
                "dup    v20.8b, v6.16b[4]            \n" // k04
                "dup    v21.8b, v6.16b[5]            \n" // k05
                "dup    v22.8b, v6.16b[6]            \n" // k06
                "dup    v23.8b, v6.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%7, #128]        \n"
                "ld1    {v26.4s}, [%7]       	     \n" // sum6
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "st1    {v26.4s}, [%7], #16  	     \n" 
                //##########################################		
                "dup    v16.8b, v7.16b[0]            \n" // k00
                "dup    v17.8b, v7.16b[1]            \n" // k01
                "dup    v18.8b, v7.16b[2]            \n" // k02
                "dup    v19.8b, v7.16b[3]            \n" // k03
                "dup    v20.8b, v7.16b[4]            \n" // k04
                "dup    v21.8b, v7.16b[5]            \n" // k05
                "dup    v22.8b, v7.16b[6]            \n" // k06
                "dup    v23.8b, v7.16b[7]            \n" // k07

                "smull  v24.8h, v8.8b, v16.8b        \n" // r0 * k0
                "smlal  v24.8h, v9.8b, v17.8b        \n" // r0 * k1
                "smlal  v24.8h, v10.8b, v18.8b       \n" // r0 * k2
                "smlal  v24.8h, v11.8b, v19.8b       \n" // r0 * k3
                "smlal  v24.8h, v12.8b, v20.8b       \n" // r0 * k4
                "smlal  v24.8h, v13.8b, v21.8b       \n" // r0 * k5
                "smlal  v24.8h, v14.8b, v22.8b       \n" // r0 * k6
                "smlal  v24.8h, v15.8b, v23.8b       \n" // r0 * k7
                
                "prfm   pldl1keep, [%8, #128]        \n"
                "ld1    {v26.4s}, [%8]       	     \n" // sum7
                "saddw  v26.4s, v26.4s, v24.4h       \n"
                "st1    {v26.4s}, [%8], #16  	     \n" 
                "sub    %18, %18, #4                 \n"
                "sub    %19, %19, #4                 \n"
                "sub    %20, %20, #4                 \n"
                "sub    %21, %21, #4                 \n"
                "sub    %22, %22, #4                 \n"
                "sub    %23, %23, #4                 \n"
                "sub    %24, %24, #4                 \n"
                "sub    %25, %25, #4                 \n"
                //##########################################					
                : "=r"(nn),     // %0
                  "=r"(outptr0),// %1
                  "=r"(outptr1),// %2
                  "=r"(outptr2),// %3
                  "=r"(outptr3),// %4
                  "=r"(outptr4),// %5
                  "=r"(outptr5),// %6
                  "=r"(outptr6),// %7
                  "=r"(outptr7) // %8
                : "0"(nn),      
                  "1"(outptr0),
                  "2"(outptr1),
                  "3"(outptr2),
                  "4"(outptr3),
                  "5"(outptr4),
                  "6"(outptr5),
                  "7"(outptr6),
                  "8"(outptr7),
                  "r"(r0),              // %18
                  "r"(r1),		// %19
                  "r"(r2),		// %20
                  "r"(r3),		// %21
                  "r"(r4),		// %22
                  "r"(r5),		// %23
                  "r"(r6),		// %24
                  "r"(r7)		// %25
                : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29"
            );                
            }
#endif	
	        for (; remain>0; remain--)
            {
                // TODO neon optimize
                short sum0 = (short)*r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3] + *r4 * kernel0[4] + *r5 * kernel0[5] + *r6 * kernel0[6] + *r7 * kernel0[7];
                short sum1 = (short)*r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3] + *r4 * kernel1[4] + *r5 * kernel1[5] + *r6 * kernel1[6] + *r7 * kernel1[7];
                short sum2 = (short)*r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3] + *r4 * kernel2[4] + *r5 * kernel2[5] + *r6 * kernel2[6] + *r7 * kernel2[7];
                short sum3 = (short)*r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3] + *r4 * kernel3[4] + *r5 * kernel3[5] + *r6 * kernel3[6] + *r7 * kernel3[7];
                short sum4 = (short)*r0 * kernel4[0] + *r1 * kernel4[1] + *r2 * kernel4[2] + *r3 * kernel4[3] + *r4 * kernel4[4] + *r5 * kernel4[5] + *r6 * kernel4[6] + *r7 * kernel4[7];
                short sum5 = (short)*r0 * kernel5[0] + *r1 * kernel5[1] + *r2 * kernel5[2] + *r3 * kernel5[3] + *r4 * kernel5[4] + *r5 * kernel5[5] + *r6 * kernel5[6] + *r7 * kernel5[7];
                short sum6 = (short)*r0 * kernel6[0] + *r1 * kernel6[1] + *r2 * kernel6[2] + *r3 * kernel6[3] + *r4 * kernel6[4] + *r5 * kernel6[5] + *r6 * kernel6[6] + *r7 * kernel6[7];
                short sum7 = (short)*r0 * kernel7[0] + *r1 * kernel7[1] + *r2 * kernel7[2] + *r3 * kernel7[3] + *r4 * kernel7[4] + *r5 * kernel7[5] + *r6 * kernel7[6] + *r7 * kernel7[7];

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;
                *outptr4 += sum4;
                *outptr5 += sum5;
                *outptr6 += sum6;
                *outptr7 += sum7;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
                outptr4++;
                outptr5++;
                outptr6++;
                outptr7++;          
            }
        }    
#endif
        for (; q<inch; q++)
        {
            short* outptr0 = out0;
            short* outptr1 = out1;
            short* outptr2 = out2;
            short* outptr3 = out3;
            short* outptr4 = out4;
            short* outptr5 = out5;
            short* outptr6 = out6;
            short* outptr7 = out7;            

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;
            const signed char* kernel1 = (const signed char*)kernel + (p+1)*inch + q;
            const signed char* kernel2 = (const signed char*)kernel + (p+2)*inch + q;
            const signed char* kernel3 = (const signed char*)kernel + (p+3)*inch + q;
            const signed char* kernel4 = (const signed char*)kernel + (p+4)*inch + q;
            const signed char* kernel5 = (const signed char*)kernel + (p+5)*inch + q;
            const signed char* kernel6 = (const signed char*)kernel + (p+6)*inch + q;
            const signed char* kernel7 = (const signed char*)kernel + (p+7)*inch + q;            

            const signed char k0 = kernel0[0];
            const signed char k1 = kernel1[0];
            const signed char k2 = kernel2[0];
            const signed char k3 = kernel3[0];
            const signed char k4 = kernel4[0];
            const signed char k5 = kernel5[0];
            const signed char k6 = kernel6[0];
            const signed char k7 = kernel7[0];            

            const signed char* r0 = img0;

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            int8x8_t _k0 = vdup_n_s8(k0);
            int8x8_t _k1 = vdup_n_s8(k1);
            int8x8_t _k2 = vdup_n_s8(k2);
            int8x8_t _k3 = vdup_n_s8(k3);
            int8x8_t _k4 = vdup_n_s8(k4);
            int8x8_t _k5 = vdup_n_s8(k5);
            int8x8_t _k6 = vdup_n_s8(k6);
            int8x8_t _k7 = vdup_n_s8(k7);           

            for (; nn>0; nn--)
            {
                int8x8_t _r0 = vld1_s8(r0);

                int16x8_t _out0  = vld1q_s16(outptr0);
                int16x8_t _out1  = vld1q_s16(outptr1);
                int16x8_t _out2  = vld1q_s16(outptr2);
                int16x8_t _out3  = vld1q_s16(outptr3);
                int16x8_t _out4  = vld1q_s16(outptr4);
                int16x8_t _out5  = vld1q_s16(outptr5);
                int16x8_t _out6  = vld1q_s16(outptr6);
                int16x8_t _out7  = vld1q_s16(outptr7);              

                int16x8_t _out0_s16 = vmull_s8(_r0, _k0);
                int16x8_t _out1_s16 = vmull_s8(_r0, _k1);
                int16x8_t _out2_s16 = vmull_s8(_r0, _k2);
                int16x8_t _out3_s16 = vmull_s8(_r0, _k3);
                int16x8_t _out4_s16 = vmull_s8(_r0, _k4);
                int16x8_t _out5_s16 = vmull_s8(_r0, _k5);
                int16x8_t _out6_s16 = vmull_s8(_r0, _k6);
                int16x8_t _out7_s16 = vmull_s8(_r0, _k7);           

                _out0  = vqaddq_s16(_out0, _out0_s16);
                _out1  = vqaddq_s16(_out1, _out1_s16);
                _out2  = vqaddq_s16(_out2, _out2_s16);
                _out3  = vqaddq_s16(_out3, _out3_s16);
                _out4  = vqaddq_s16(_out4, _out4_s16);
                _out5  = vqaddq_s16(_out5, _out5_s16);
                _out6  = vqaddq_s16(_out6, _out6_s16);
                _out7  = vqaddq_s16(_out7, _out7_s16);

                vst1q_s16(outptr0, _out0);
                vst1q_s16(outptr1, _out1);
                vst1q_s16(outptr2, _out2);
                vst1q_s16(outptr3, _out3);
                vst1q_s16(outptr4, _out4);
                vst1q_s16(outptr5, _out5);
                vst1q_s16(outptr6, _out6);
                vst1q_s16(outptr7, _out7);

                r0 += 8;
                outptr0 += 8;
                outptr1 += 8;
                outptr2 += 8;
                outptr3 += 8;
                outptr4 += 8;
                outptr5 += 8;
                outptr6 += 8;
                outptr7 += 8;                
            }
            
            for (; remain>0; remain--)
            {
                // TODO neon optimize
                short sum0 = (short)*r0 * k0;
                short sum1 = (short)*r0 * k1;
                short sum2 = (short)*r0 * k2;
                short sum3 = (short)*r0 * k3;
                short sum4 = (short)*r0 * k4;
                short sum5 = (short)*r0 * k5;
                short sum6 = (short)*r0 * k6;
                short sum7 = (short)*r0 * k7;              

                *outptr0 += sum0;
                *outptr1 += sum1;
                *outptr2 += sum2;
                *outptr3 += sum3;
                *outptr4 += sum4;
                *outptr5 += sum5;
                *outptr6 += sum6;
                *outptr7 += sum7;                

                r0++;
                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
                outptr4++;
                outptr5++;
                outptr6++;
                outptr7++;                
            }
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        out.fill(0);

        int q = 0;

        for (; q+7<inch; q+=8)
        {
            short* outptr = out;

            const signed char* img0 = bottom_blob.channel(q);
            const signed char* img1 = bottom_blob.channel(q+1);
            const signed char* img2 = bottom_blob.channel(q+2);
            const signed char* img3 = bottom_blob.channel(q+3);
            const signed char* img4 = bottom_blob.channel(q+4);
            const signed char* img5 = bottom_blob.channel(q+5);
            const signed char* img6 = bottom_blob.channel(q+6);
            const signed char* img7 = bottom_blob.channel(q+7);            

            const signed char* kernel0 = (const signed char*)kernel + p*inch + q;
            const signed char k0 = kernel0[0];
            const signed char k1 = kernel0[1];
            const signed char k2 = kernel0[2];
            const signed char k3 = kernel0[3];
            const signed char k4 = kernel0[4];
            const signed char k5 = kernel0[5];
            const signed char k6 = kernel0[6];
            const signed char k7 = kernel0[7];            

            const signed char* r0 = img0;
            const signed char* r1 = img1;
            const signed char* r2 = img2;
            const signed char* r3 = img3;
            const signed char* r4 = img4;
            const signed char* r5 = img5;
            const signed char* r6 = img6;
            const signed char* r7 = img7;            

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            int8x8_t _k0 = vdup_n_s8(k0);
            int8x8_t _k1 = vdup_n_s8(k1);
            int8x8_t _k2 = vdup_n_s8(k2);
            int8x8_t _k3 = vdup_n_s8(k3);
            int8x8_t _k4 = vdup_n_s8(k4);
            int8x8_t _k5 = vdup_n_s8(k5);
            int8x8_t _k6 = vdup_n_s8(k6);
            int8x8_t _k7 = vdup_n_s8(k7);            

            for (; nn>0; nn--)
            {
                int8x8_t _r0 = vld1_s8(r0);
                int8x8_t _r1 = vld1_s8(r1);
                int8x8_t _r2 = vld1_s8(r2);
                int8x8_t _r3 = vld1_s8(r3);
                int8x8_t _r4 = vld1_s8(r4);
                int8x8_t _r5 = vld1_s8(r5);
                int8x8_t _r6 = vld1_s8(r6);
                int8x8_t _r7 = vld1_s8(r7);

                int16x8_t _out0 = vld1q_s16(outptr);

                int16x8_t _out0_s16 = vmull_s8(_r0, _k0);
                _out0_s16 = vmlal_s8(_out0_s16, _r1, _k1);
                _out0_s16 = vmlal_s8(_out0_s16, _r2, _k2);
                _out0_s16 = vmlal_s8(_out0_s16, _r3, _k3);
                _out0_s16 = vmlal_s8(_out0_s16, _r4, _k4);
                _out0_s16 = vmlal_s8(_out0_s16, _r5, _k5);
                _out0_s16 = vmlal_s8(_out0_s16, _r6, _k6);
                _out0_s16 = vmlal_s8(_out0_s16, _r7, _k7);

                _out0 = vqaddq_s16(_out0, _out0_s16);

                vst1q_s16(outptr, _out0);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                r4 += 8;
                r5 += 8;
                r6 += 8;
                r7 += 8;
                outptr += 8;         
            }

            for (; remain>0; remain--)
            {
                short sum  = (short)*r0 * k0;
                short sum1 = (short)*r1 * k1;
                short sum2 = (short)*r2 * k2;
                short sum3 = (short)*r3 * k3;
                short sum4 = (short)*r4 * k4;
                short sum5 = (short)*r5 * k5;
                short sum6 = (short)*r6 * k6;
                short sum7 = (short)*r7 * k7;                

                *outptr += sum + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;                
                outptr++;
            }

        }

        for (; q<inch; q++)
        {
            short* outptr = out;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* kernel0 = (const signed char*)kernel + p*inch  + q;
            const signed char k0 = kernel0[0];

            const signed char* r0 = img0;

            int size = outw * outh;

            int nn = size >> 3;
            int remain = size & 7;

            int8x8_t _k0 = vdup_n_s8(k0);

            for (; nn>0; nn--)
            {
                int8x8_t _r0 = vld1_s8(r0);

                int16x8_t _out0 = vld1q_s16(outptr);

                int16x8_t _out0_s16 = vmull_s8(_r0, _k0);

                _out0 = vqaddq_s16(_out0, _out0_s16);

                vst1q_s16(outptr, _out0);

                r0 += 8;
                outptr += 8;
            }

            for (; remain>0; remain--)
            {
                short sum = (short)*r0 * k0;

                *outptr += sum;

                r0++;
                outptr++;
            }
        }
    }    
}


static void conv1x1s1_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int kernel_w = 1;
    int kernel_h = 1;

    int stride_w = 1;
    int stride_h = 1;

    conv_im2col_sgemm_int8_neon(bottom_blob, top_blob, _kernel, kernel_w, kernel_h, stride_w, stride_h, opt);
}

static void conv1x1s2_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int kernel_w = 1;
    int kernel_h = 1;

    int stride_w = 2;
    int stride_h = 2;

    conv_im2col_sgemm_int8_neon(bottom_blob, top_blob, _kernel, kernel_w, kernel_h, stride_w, stride_h, opt);
}


static void conv1x1s1_int8_dequant_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Mat &_bias, std::vector<float> scales_dequant, const Option& opt)
{
    int kernel_w = 1;
    int kernel_h = 1;

    int stride_w = 1;
    int stride_h = 1;

    conv_im2col_sgemm_int8_dequant_neon(bottom_blob, top_blob, _kernel, kernel_w, kernel_h, stride_w, stride_h, _bias, scales_dequant, opt);
}

static void conv1x1s2_int8_dequant_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Mat &_bias, std::vector<float> scales_dequant, const Option& opt)
{
    int kernel_w = 1;
    int kernel_h = 1;

    int stride_w = 2;
    int stride_h = 2;

    conv_im2col_sgemm_int8_dequant_neon(bottom_blob, top_blob, _kernel, kernel_w, kernel_h, stride_w, stride_h, _bias, scales_dequant, opt);
}

static void conv1x1s1_int8_requant_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Mat &_bias, std::vector<float> scales_requant, const Option& opt)
{
    int kernel_w = 1;
    int kernel_h = 1;

    int stride_w = 1;
    int stride_h = 1;

    conv_im2col_sgemm_int8_requant_neon(bottom_blob, top_blob, _kernel, kernel_w, kernel_h, stride_w, stride_h, _bias, scales_requant, opt);
}

static void conv1x1s2_int8_requant_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Mat &_bias, std::vector<float> scales_requant, const Option& opt)
{
    int kernel_w = 1;
    int kernel_h = 1;

    int stride_w = 2;
    int stride_h = 2;

    conv_im2col_sgemm_int8_requant_neon(bottom_blob, top_blob, _kernel, kernel_w, kernel_h, stride_w, stride_h, _bias, scales_requant, opt);
}