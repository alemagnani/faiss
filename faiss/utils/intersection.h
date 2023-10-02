//
// Created by alessandro on 9/20/23.
//

#ifndef FAISS_INTERSECTION_H
#define FAISS_INTERSECTION_H


#include <stdint.h>
#include <iostream>
#include <map>
#include <vector>

namespace faiss {

using namespace std;

// Taken from https://github.com/lemire/SIMDCompressionAndIntersection


size_t SIMDintersection(const uint32_t *set1, const size_t length1,
                        const uint32_t *set2, const size_t length2,
                        uint32_t *out);



size_t scalar(const uint32_t *set1, const size_t length1,
                        const uint32_t *set2, const size_t length2,
                        uint32_t *out);



/*
size_t highlyscalable_intersect_SIMD(const uint32_t *set1, const size_t length1,
                                     const uint32_t *set2, const size_t length2,
                                     uint32_t *out);
*/


}
#endif // FAISS_INTERSECTION_H
