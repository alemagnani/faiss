/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>

namespace faiss {

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


void IDSelectorIVFTwo::set_list(idx_t list_no) const{

    this->sel1->set_list(list_no);
    if (this -> sel2 != nullptr) {
        this->sel2->set_list(list_no);
    }
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

bool find_cluster(const int32_t n, const int16_t* array, const idx_t list_no, int32_t  &found_pos) {

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
    if (this->sel1 == nullptr or (limit_w2_low >=0  && this->sel2 == nullptr)) {
            return false;
    }
    const bool out = this->sel1->is_member(id);
    if ( !out or limit_w2_low < 0 ){
        return out;
    }
    return this->sel2->is_member(id);
}


void IDSelectorIVFClusterAware::set_list(idx_t list_no) const{

    delete(this->sel1);
    delete(this->sel2);

    int32_t cluster_pos;
    bool found;
    found = find_cluster(limit_w1_high-limit_w1_low, clusters + limit_w1_low, list_no, cluster_pos );
    //printf("finding for %ld list, %d begin %d end\n", list_no, begin, end);
    int32_t ids_pos_start;
    int32_t ids_pos_end;
    if ( found ) {
        ids_pos_start = limits[cluster_pos], ids_pos_end = limits[cluster_pos + 1];
        this->sel1 = new IDSelectorIVFSingle(
                (ids_pos_end - ids_pos_start), this->ids + ids_pos_start);
    }
    if ( limit_w2_low >=0 ) {
        found = find_cluster(limit_w2_high-limit_w2_low, clusters + limit_w2_low, list_no, cluster_pos );
        if ( found ) {
                ids_pos_start = limits[cluster_pos], ids_pos_end = limits[cluster_pos + 1];
                this->sel2 = new IDSelectorIVFSingle(
                        (ids_pos_end - ids_pos_start),
                        this->ids + ids_pos_start);
        }
    }
}






} // namespace faiss
