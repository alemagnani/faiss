/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/intersection.h>

#include <cstring>


namespace faiss {


uint64_t get_my_cy() {
    uint32_t high, low;
    asm volatile("rdtsc \n\t" : "=a"(low), "=d"(high));
    return ((uint64_t)high << 32) | (low);
}


/***********************************************************************
 * IDSelectorRange
 ***********************************************************************/

IDSelectorRange::IDSelectorRange(idx_t imin, idx_t imax, bool assume_sorted)
        : imin(imin), imax(imax), assume_sorted(assume_sorted) {}

bool IDSelectorRange::is_member(idx_t id) const {
    return id >= imin && id < imax;
}

void IDSelectorRange::find_sorted_ids_bounds(
        size_t list_size,
        const idx_t* ids,
        size_t* jmin_out,
        size_t* jmax_out) const {
    FAISS_ASSERT(assume_sorted);
    if (list_size == 0 || imax <= ids[0] || imin > ids[list_size - 1]) {
        *jmin_out = *jmax_out = 0;
        return;
    }
    // bissection to find imin
    if (ids[0] >= imin) {
        *jmin_out = 0;
    } else {
        size_t j0 = 0, j1 = list_size;
        while (j1 > j0 + 1) {
            size_t jmed = (j0 + j1) / 2;
            if (ids[jmed] >= imin) {
                j1 = jmed;
            } else {
                j0 = jmed;
            }
        }
        *jmin_out = j1;
    }
    // bissection to find imax
    if (*jmin_out == list_size || ids[*jmin_out] >= imax) {
        *jmax_out = *jmin_out;
    } else {
        size_t j0 = *jmin_out, j1 = list_size;
        while (j1 > j0 + 1) {
            size_t jmed = (j0 + j1) / 2;
            if (ids[jmed] >= imax) {
                j1 = jmed;
            } else {
                j0 = jmed;
            }
        }
        *jmax_out = j1;
    }
}

/***********************************************************************
 * IDSelectorArray
 ***********************************************************************/

IDSelectorArray::IDSelectorArray(size_t n, const idx_t* ids) : n(n), ids(ids) {}

bool IDSelectorArray::is_member(idx_t id) const {
    for (idx_t i = 0; i < n; i++) {
        if (ids[i] == id)
            return true;
    }
    return false;
}

/***********************************************************************
 * IDSelectorBatch
 ***********************************************************************/

IDSelectorBatch::IDSelectorBatch(size_t n, const idx_t* indices) {
    nbits = 0;
    while (n > ((idx_t)1 << nbits)) {
        nbits++;
    }
    nbits += 5;
    // for n = 1M, nbits = 25 is optimal, see P56659518

    mask = ((idx_t)1 << nbits) - 1;
    bloom.resize((idx_t)1 << (nbits - 3), 0);
    for (idx_t i = 0; i < n; i++) {
        idx_t id = indices[i];
        set.insert(id);
        id &= mask;
        bloom[id >> 3] |= 1 << (id & 7);
    }
}

bool IDSelectorBatch::is_member(idx_t i) const {
    long im = i & mask;
    if (!(bloom[im >> 3] & (1 << (im & 7)))) {
        return 0;
    }
    return set.count(i);
}

/***********************************************************************
 * IDSelectorBitmap
 ***********************************************************************/

IDSelectorBitmap::IDSelectorBitmap(size_t n, const uint8_t* bitmap)
        : n(n), bitmap(bitmap) {}

bool IDSelectorBitmap::is_member(idx_t ii) const {
    uint64_t i = ii;
    if ((i >> 3) >= n) {
        return false;
    }
    return (bitmap[i >> 3] >> (i & 7)) & 1;
}

/***********************************************************************
 * IDSelectorIVF
 ***********************************************************************/


IDSelectorIVFSingle::IDSelectorIVFSingle(size_t n, const int32_t* ids) : n(n), ids(ids) {};

bool IDSelectorIVFSingle::is_member(idx_t id) const{
    if ( n == 0 ) {
        return false;
    }
    int32_t value;
    while(offset < n) {
        value = ids[offset];
        //printf("%d, %ld, %d\n", value, id, offset);
        if ( value > id ){
            return false;
        } else if ( value == id) {
            offset++;
            return true;
        }
        offset++;
    }
    return false;
}


IDSelectorIVFTwo::IDSelectorIVFTwo(const int32_t* ids, const int32_t* limits) :  ids(ids), limits(limits){};


void IDSelectorIVFTwo::set_words(int32_t w1, int32_t w2) {

    delete(this->sel1);
    delete(this->sel2);

    this->sel1 = new IDSelectorIVFSingle(limits[w1+1]-limits[w1], this->ids + limits[w1]);
    if ( w2 >= 0) {
        this->sel2 = new IDSelectorIVFSingle(limits[w2+1]-limits[w2], this->ids + limits[w2]);
    } else {
        this->sel2 = nullptr;
    }
}


bool IDSelectorIVFTwo::set_list(idx_t list_no) const{

    this->sel1->set_list(list_no);
    if (this -> sel2 != nullptr) {
        this->sel2->set_list(list_no);
    }
    return true;
}


bool IDSelectorIVFTwo::is_member(idx_t id) const  {
    const bool out = this->sel1->is_member(id);
    if ( !out or this-> sel2 == nullptr){
        return out;
    }
    return this->sel2->is_member(id);
}

/***********************************************************************
 * IDSelectorIVFClusterAware
 ***********************************************************************/

IDSelectorIVFClusterAware::IDSelectorIVFClusterAware(const int32_t* ids,  const int32_t* limits, const int16_t* clusters, const int32_t* cluster_limits) :  ids(ids), clusters(clusters), cluster_limits(cluster_limits), limits(limits){};

void IDSelectorIVFClusterAware::set_words(int32_t w1, int32_t w2) {
    //printf("set words %ld,  %ld\n", w1, w2);
    limit_w1_low = cluster_limits[w1];
    limit_w1_high = cluster_limits[w1+1];
    if ( w2 >= 0 ) {
        limit_w2_low = cluster_limits[w2];
        limit_w2_high = cluster_limits[w2+1];
    } else {
        limit_w2_low = -1;
        limit_w2_high = -1;
    }
}

inline bool find_cluster(const int32_t n, const int16_t* array, const int_fast16_t list_no, int32_t  &found_pos) {

    if (n == 0 || array[0] > list_no || array[n-1] < list_no ) {
        return false;
    }
    if ( array[0] == list_no ) {
        found_pos = 0;
        return true;
    }
    int32_t j0 = 0, j1 = n;

    while (j1 > j0 + 1) {
            int32_t jmed = (j0 + j1) / 2;
            if (array[jmed] > list_no) {
                j1 = jmed;
            }
            else if ( array[jmed] == list_no) {
                found_pos = jmed;
                return true;
            } else {
                j0 = jmed;
            }
    }
    return false;
}

bool IDSelectorIVFClusterAware::is_member(idx_t id) const  {
    if ( !found_cluster_w1 or (limit_w2_low >=0  && !found_cluster_w2)) {
            return false;
    }
    const bool out = this->sel1->is_member(id);
    if ( !out or limit_w2_low < 0 ){
        return out;
    }
    return this->sel2->is_member(id);
}


bool IDSelectorIVFClusterAware::set_list(idx_t list_no) const{

    delete(this->sel1);
    this->sel1 = nullptr;
    delete(this->sel2);
    this->sel2 = nullptr;

    int32_t cluster_pos;

    found_cluster_w1 = find_cluster(limit_w1_high-limit_w1_low, clusters + limit_w1_low, list_no, cluster_pos );

    int32_t ids_pos_start;
    int32_t ids_pos_end;
    if ( found_cluster_w1 ) {
        ids_pos_start = limits[limit_w1_low + cluster_pos], ids_pos_end = limits[limit_w1_low + cluster_pos + 1];
        this->sel1 = new IDSelectorIVFSingle(
                (ids_pos_end - ids_pos_start), this->ids + ids_pos_start);
    }
    if ( limit_w2_low >=0 ) {
        found_cluster_w2 = find_cluster(limit_w2_high-limit_w2_low, clusters + limit_w2_low, list_no, cluster_pos );
        if ( found_cluster_w2 ) {
                ids_pos_start = limits[limit_w2_low + cluster_pos], ids_pos_end = limits[limit_w2_low + cluster_pos + 1];
                this->sel2 = new IDSelectorIVFSingle(
                        (ids_pos_end - ids_pos_start),
                        this->ids + ids_pos_start);
        } else {
                found_cluster_w1 = false;
        }
    }
    return true;
}

/***********************************************************************
 * IDSelectorIVFClusterAwareIntersect
 ***********************************************************************/

IDSelectorIVFClusterAwareIntersect::IDSelectorIVFClusterAwareIntersect(const int32_t* ids,  const int32_t* limits, const int16_t* clusters, const int32_t* cluster_limits, idx_t size_tmp) :  ids(ids), clusters(clusters), cluster_limits(cluster_limits), limits(limits){
                                                                                tmp = new int32_t [size_tmp];
                                                                        };

void IDSelectorIVFClusterAwareIntersect::set_words(int32_t w1, int32_t w2) {
    //printf("set words %ld,  %ld\n", w1, w2);
    limit_w1_low = cluster_limits[w1];
    limit_w1_high = cluster_limits[w1+1];
    if ( w2 >= 0 ) {
        limit_w2_low = cluster_limits[w2];
        limit_w2_high = cluster_limits[w2+1];
    } else {
        limit_w2_low = -1;
        limit_w2_high = -1;
    }
}


bool IDSelectorIVFClusterAwareIntersect::is_member(idx_t id) const  {
    //printf("checking memebership in intersect for id %ld\n", id);
    if ( !found_cluster_w){
        return false;
    }
    return  this->sel->is_member(id);
}


bool IDSelectorIVFClusterAwareIntersect::set_list(idx_t list_no) const{

    //printf("set list in intersect inernal %ld\n",list_no);
    delete(this->sel);
    this->sel = nullptr;
    ///printf("done\n");
    int32_t cluster_pos;

    found_cluster_w = find_cluster(limit_w1_high-limit_w1_low, clusters + limit_w1_low, list_no, cluster_pos );

    int32_t ids_pos_start;
    int32_t ids_pos_end;
    if ( found_cluster_w ) {
        ids_pos_start = limits[limit_w1_low + cluster_pos], ids_pos_end = limits[limit_w1_low + cluster_pos + 1];
        if (limit_w2_low < 0) {
                this->sel = new IDSelectorIVFSingle(
                        (ids_pos_end - ids_pos_start),
                        this->ids + ids_pos_start);
                return (ids_pos_end - ids_pos_start) > 0;
        } else {
                found_cluster_w = find_cluster(limit_w2_high-limit_w2_low, clusters + limit_w2_low, list_no, cluster_pos );
                if ( found_cluster_w ) {
                    int32_t ids_pos_start_2;
                    int32_t ids_pos_end_2;
                    ids_pos_start_2 = limits[limit_w2_low + cluster_pos], ids_pos_end_2 = limits[limit_w2_low + cluster_pos + 1];

                    //printf("intersecting size %d, %d \n", ids_pos_end - ids_pos_start, ids_pos_end_2 - ids_pos_start_2);

                    size_t size_intersection =  SIMDintersection(
                            reinterpret_cast<const uint32_t*>(
                                    this->ids + ids_pos_start), ids_pos_end - ids_pos_start,
                            reinterpret_cast<const uint32_t*>(
                                    this->ids + ids_pos_start_2), ids_pos_end_2 - ids_pos_start_2,
                            reinterpret_cast<uint32_t*>(this->tmp));


                    //printf("intersecting size %d, %d and output intersection is %d\n", ids_pos_end - ids_pos_start, ids_pos_end_2 - ids_pos_start_2, size_intersection);
                    if (size_intersection == 0 ) {
                        found_cluster_w = false;
                        return false;
                    }
                    this->sel = new IDSelectorIVFSingle(
                            size_intersection,
                            this->tmp);
                    return  true;
                } else {
                    return false;
                }
        }
    } else {
        return false;
    }
}

/***********************************************************************
 * IDSelectorIVFClusterAwareIntersectDirect
 ***********************************************************************/

IDSelectorIVFClusterAwareIntersectDirect::IDSelectorIVFClusterAwareIntersectDirect(const int32_t* in_cluster_positions,  const int32_t* limits, const int16_t* clusters, const int32_t* cluster_limits,  idx_t size_tmp) :  in_cluster_positions(in_cluster_positions), clusters(clusters), cluster_limits(cluster_limits), limits(limits){
    tmp = new int32_t [size_tmp];
};

void IDSelectorIVFClusterAwareIntersectDirect::set_words(int32_t w1, int32_t w2) {
    //printf("set words %ld,  %ld\n", w1, w2);
    limit_w1_low = cluster_limits[w1];
    limit_w1_high = cluster_limits[w1+1];
    if ( w2 >= 0 ) {
        limit_w2_low = cluster_limits[w2];
        limit_w2_high = cluster_limits[w2+1];
    } else {
        limit_w2_low = -1;
        limit_w2_high = -1;
    }
}


bool IDSelectorIVFClusterAwareIntersectDirect::is_member(idx_t id) const  {
    // we are not using this method for this
    return  false;
}


bool IDSelectorIVFClusterAwareIntersectDirect::set_list(idx_t list_no) const{
    //uint64_t start = get_my_cy();
    int32_t cluster_pos;

    //printf("settingn correctly\n");

    bool found_cluster_w = find_cluster(limit_w1_high-limit_w1_low, clusters + limit_w1_low, list_no, cluster_pos );
    //uint64_t end_find_cluster = get_my_cy();
    //IDSelectorMy_Stats.find_cluster += end_find_cluster - start;

    //printf("set list\n");
    int32_t ids_pos_start;
    int32_t ids_pos_end;
    if ( found_cluster_w ) {
        ids_pos_start = limits[limit_w1_low + cluster_pos], ids_pos_end = limits[limit_w1_low + cluster_pos + 1];
        if (limit_w2_low < 0) {
                //printf("found cluster of size %d\n", ids_pos_end - ids_pos_start );
                this->range = this->in_cluster_positions + ids_pos_start;
                this->range_size = ids_pos_end - ids_pos_start;

                //uint64_t end = get_my_cy();
                //IDSelectorMy_Stats.set_list_time += end - start;
                return this->range_size > 0;
        } else {

                //uint64_t start_find_cluster = get_my_cy();
                found_cluster_w = find_cluster(limit_w2_high-limit_w2_low, clusters + limit_w2_low, list_no, cluster_pos );

                //end_find_cluster = get_my_cy();
                //IDSelectorMy_Stats.find_cluster += end_find_cluster - start_find_cluster;

                if ( found_cluster_w ) {
                    int32_t ids_pos_start_2;
                    int32_t ids_pos_end_2;
                    ids_pos_start_2 = limits[limit_w2_low + cluster_pos], ids_pos_end_2 = limits[limit_w2_low + cluster_pos + 1];

                    //uint64_t start_intersection = get_my_cy();
                    size_t size_intersection =  SIMDintersection(
                            reinterpret_cast<const uint32_t*>(
                                    this->in_cluster_positions + ids_pos_start), ids_pos_end - ids_pos_start,
                            reinterpret_cast<const uint32_t*>(
                                    this->in_cluster_positions + ids_pos_start_2), ids_pos_end_2 - ids_pos_start_2,
                            reinterpret_cast<uint32_t*>(this->tmp));
                    //uint64_t end_intersection = get_my_cy();
                    //IDSelectorMy_Stats.intersection += end_intersection - start_intersection;

                    this->range = this->tmp;
                    this->range_size = size_intersection;
                    //printf("found size of intersection %d\n",size_intersection);
                    //uint64_t end = get_my_cy();
                    //IDSelectorMy_Stats.set_list_time += end - start;
                    return size_intersection > 0;

                } else {
                    this->range = nullptr;
                    this->range_size = 0;
                    //uint64_t end = get_my_cy();
                    //IDSelectorMy_Stats.set_list_time += end - start;
                    return false;
                }
        }
    } else {
        this->range = nullptr;
        this->range_size = 0;
        ///uint64_t end = get_my_cy();
        //IDSelectorMy_Stats.set_list_time += end - start;
        return false;
    }
    uint64_t end = get_my_cy();
    //IDSelectorMy_Stats.set_list_time += end - start;
    return true;
}

/***********************************************************************
 * IDSelectorIVFClusterAwareIntersectDirectExp
 ***********************************************************************/

IDSelectorIVFClusterAwareIntersectDirectExp::IDSelectorIVFClusterAwareIntersectDirectExp(const int32_t* in_cluster_positions, const int32_t * limits, int32_t nprobes, idx_t size_tmp) :  in_cluster_positions(in_cluster_positions), limits(limits), nprobes(nprobes){
    tmp = new int32_t [size_tmp];
};

void IDSelectorIVFClusterAwareIntersectDirectExp::set_words(int32_t w1_l, int32_t w2_l) {
    //printf("set words %ld,  %ld\n", w1, w2);
    w1 = w1_l * nprobes;
    if ( w2_l >= 0 ) {
       w2 = w2_l * nprobes;
    } else {
        w2 = -1;
    }
}


bool IDSelectorIVFClusterAwareIntersectDirectExp::is_member(idx_t id) const  {
    // we are not using this method for this
    return  false;
}


bool IDSelectorIVFClusterAwareIntersectDirectExp::set_list(idx_t list_no) const{

    int32_t ids_pos_start = limits[w1 + list_no];
    int32_t ids_pos_end = limits[w1 + list_no + 1];
    if ( ids_pos_start >=0 && ids_pos_end >=0 && ids_pos_end > ids_pos_start ) {

        if (w2 < 0) {
                //printf("found cluster of size %d\n", ids_pos_end - ids_pos_start );
                this->range = this->in_cluster_positions + ids_pos_start;
                this->range_size = ids_pos_end - ids_pos_start;
                return true;
        } else {

                int32_t ids_pos_start_2 = limits[w2 + list_no];
                int32_t ids_pos_end_2 = limits[w2 + list_no + 1];

                if ( ids_pos_start_2 >=0 && ids_pos_end_2 >=0 && ids_pos_end_2 > ids_pos_start_2 ) {

                    size_t size_intersection =  SIMDintersection(
                            reinterpret_cast<const uint32_t*>(
                                    this->in_cluster_positions + ids_pos_start), ids_pos_end - ids_pos_start,
                            reinterpret_cast<const uint32_t*>(
                                    this->in_cluster_positions + ids_pos_start_2), ids_pos_end_2 - ids_pos_start_2,
                            reinterpret_cast<uint32_t*>(this->tmp));

                    this->range = this->tmp;
                    this->range_size = size_intersection;

                    return size_intersection > 0;

                } else {
                    this->range = nullptr;
                    this->range_size = 0;
                    return false;
                }
        }
    } else {
        this->range = nullptr;
        this->range_size = 0;
        return false;
    }
}



void IDSelectorMyStats::reset() {
    memset((void*)this, 0, sizeof(*this));
}

//IDSelectorMyStats IDSelectorMy_Stats;


} // namespace faiss
