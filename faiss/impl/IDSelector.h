/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_set>
#include <vector>

#include <faiss/Index.h>

/** IDSelector is intended to define a subset of vectors to handle (for removal
 * or as subset to search) */

namespace faiss {

/** Encapsulates a set of ids to handle. */
struct IDSelector {
    virtual bool is_member(idx_t id) const = 0;
    virtual ~IDSelector() {}
};

/** ids between [imin, imax) */
struct IDSelectorRange : IDSelector {
    idx_t imin, imax;

    /// Assume that the ids to handle are sorted. In some cases this can speed
    /// up processing
    bool assume_sorted;

    IDSelectorRange(idx_t imin, idx_t imax, bool assume_sorted = false);

    bool is_member(idx_t id) const final;

    /// for sorted ids, find the range of list indices where the valid ids are
    /// stored
    void find_sorted_ids_bounds(
            size_t list_size,
            const idx_t* ids,
            size_t* jmin,
            size_t* jmax) const;

    ~IDSelectorRange() override {}
};

/** Simple array of elements
 *
 * is_member calls are very inefficient, but some operations can use the ids
 * directly.
 */
struct IDSelectorArray : IDSelector {
    size_t n;
    const idx_t* ids;

    /** Construct with an array of ids to process
     *
     * @param n number of ids to store
     * @param ids elements to store. The pointer should remain valid during
     *            IDSelectorArray's lifetime
     */
    IDSelectorArray(size_t n, const idx_t* ids);
    bool is_member(idx_t id) const final;
    ~IDSelectorArray() override {}
};

/** Ids from a set.
 *
 * Repetitions of ids in the indices set passed to the constructor does not hurt
 * performance.
 *
 * The hash function used for the bloom filter and GCC's implementation of
 * unordered_set are just the least significant bits of the id. This works fine
 * for random ids or ids in sequences but will produce many hash collisions if
 * lsb's are always the same
 */
struct IDSelectorBatch : IDSelector {
    std::unordered_set<idx_t> set;

    // Bloom filter to avoid accessing the unordered set if it is unlikely
    // to be true
    std::vector<uint8_t> bloom;
    int nbits;
    idx_t mask;

    /** Construct with an array of ids to process
     *
     * @param n number of ids to store
     * @param ids elements to store. The pointer can be released after
     *            construction
     */
    IDSelectorBatch(size_t n, const idx_t* indices);
    bool is_member(idx_t id) const final;
    ~IDSelectorBatch() override {}
};

/** One bit per element. Constructed with a bitmap, size ceil(n / 8).
 */
struct IDSelectorBitmap : IDSelector {
    size_t n;
    const uint8_t* bitmap;

    /** Construct with a binary mask
     *
     * @param n size of the bitmap array
     * @param bitmap id will be selected iff id / 8 < n and bit number
     *               (i%8) of bitmap[floor(i / 8)] is 1.
     */
    IDSelectorBitmap(size_t n, const uint8_t* bitmap);
    bool is_member(idx_t id) const final;
    ~IDSelectorBitmap() override {}
};

/** reverts the membership test of another selector */
struct IDSelectorNot : IDSelector {
    const IDSelector* sel;
    IDSelectorNot(const IDSelector* sel) : sel(sel) {}
    bool is_member(idx_t id) const final {
        return !sel->is_member(id);
    }
    virtual ~IDSelectorNot() {}
};

/// selects all entries (useful for benchmarking)
struct IDSelectorAll : IDSelector {
    bool is_member(idx_t id) const final {
        return true;
    }
    virtual ~IDSelectorAll() {}
};

/// does an AND operation on the the two given IDSelector's is_membership
/// results.
struct IDSelectorAnd : IDSelector {
    const IDSelector* lhs;
    const IDSelector* rhs;
    IDSelectorAnd(const IDSelector* lhs, const IDSelector* rhs)
            : lhs(lhs), rhs(rhs) {}
    bool is_member(idx_t id) const final {
        return lhs->is_member(id) && rhs->is_member(id);
    };
    virtual ~IDSelectorAnd() {}
};

/// does an OR operation on the the two given IDSelector's is_membership
/// results.
struct IDSelectorOr : IDSelector {
    const IDSelector* lhs;
    const IDSelector* rhs;
    IDSelectorOr(const IDSelector* lhs, const IDSelector* rhs)
            : lhs(lhs), rhs(rhs) {}
    bool is_member(idx_t id) const final {
        return lhs->is_member(id) || rhs->is_member(id);
    };
    virtual ~IDSelectorOr() {}
};

/// does an XOR operation on the the two given IDSelector's is_membership
/// results.
struct IDSelectorXOr : IDSelector {
    const IDSelector* lhs;
    const IDSelector* rhs;
    IDSelectorXOr(const IDSelector* lhs, const IDSelector* rhs)
            : lhs(lhs), rhs(rhs) {}
    bool is_member(idx_t id) const final {
        return lhs->is_member(id) ^ rhs->is_member(id);
    };
    virtual ~IDSelectorXOr() {}
};

struct IDSelectorIVF : IDSelector {
    // initialize the selector for the specified list
    virtual void set_list(idx_t list_no) const {};

};


struct IDSelectorIVFSingle : IDSelectorIVF {

    size_t n;
    const int32_t* ids;
    mutable int32_t  offset = 0;

    IDSelectorIVFSingle(size_t n, const int32_t* ids);

    // initialize the selector for the specified list
    void set_list(idx_t list_no) const  override {
        offset = 0;
    };

    bool is_member(idx_t id) const override;
    ~IDSelectorIVFSingle() override = default;
};




struct IDSelectorIVFTwo : IDSelectorIVF {

    const int32_t* ids;
    const int32_t * limits;
    IDSelectorIVFSingle* sel1 = nullptr;
    IDSelectorIVFSingle* sel2 = nullptr;

    IDSelectorIVFTwo(const int32_t* ids, const int32_t* limits);

    void set_words(int32_t w1, int32_t w2=-1);

    // initialize the selector for the specified list
    void set_list(idx_t list_no) const override;

    bool is_member(idx_t id) const override;

    ~IDSelectorIVFTwo() override = default;
};


struct IDSelectorIVFClusterAware : IDSelectorIVF {

    // the siZe if ids and cluster_ids is the same
    // the ids are ordered in such a way that for a given word the cluster are in order
    const int32_t* ids;
    const int32_t * limits;
    const int16_t* clusters;
    const int32_t* cluster_limits;



    int32_t limit_w1_low = -1;
    int32_t limit_w1_high = -1;
    int32_t limit_w2_low = -1;
    int32_t limit_w2_high = -1;

    mutable bool found_cluster_w1=false;
    mutable bool found_cluster_w2=false;

    mutable IDSelectorIVFSingle* sel1 = nullptr;
    mutable IDSelectorIVFSingle* sel2 = nullptr;

    IDSelectorIVFClusterAware(const int32_t* ids, const int32_t* limits, const int16_t* clusters, const int32_t* cluster_limits);

    void set_words(int32_t w1, int32_t w2=-1);

    // initialize the selector for the specified list
    void set_list(idx_t list_no) const override;

    bool is_member(idx_t id) const override;

    ~IDSelectorIVFClusterAware() override = default;
};

struct IDSelectorIVFClusterAwareIntersect : IDSelectorIVF {

    // the siZe if ids and cluster_ids is the same
    // the ids are ordered in such a way that for a given word the cluster are in order
    const int32_t* ids;
    const int32_t * limits;
    const int16_t* clusters;
    const int32_t* cluster_limits;

    int32_t* tmp;



    int32_t limit_w1_low = -1;
    int32_t limit_w1_high = -1;
    int32_t limit_w2_low = -1;
    int32_t limit_w2_high = -1;

    mutable bool found_cluster_w=false;

    mutable IDSelectorIVFSingle* sel = nullptr;


    IDSelectorIVFClusterAwareIntersect(const int32_t* ids, const int32_t* limits, const int16_t* clusters, const int32_t* cluster_limits);

    void set_words(int32_t w1, int32_t w2=-1);

    // initialize the selector for the specified list
    void set_list(idx_t list_no) const override;

    bool is_member(idx_t id) const override;

    ~IDSelectorIVFClusterAwareIntersect(

            ) override {
        delete(tmp);
    };
};

struct IDSelectorIVFClusterAwareIntersectDirect : IDSelectorIVF {

    // the siZe if ids and cluster_ids is the same
    // the ids are ordered in such a way that for a given word the cluster are in order
    const int32_t* in_cluster_positions;
    const int32_t * limits;
    const int16_t* clusters;
    const int32_t* cluster_limits;

    int32_t* tmp;

    mutable const int32_t* range;
    mutable int32_t range_size;


    int32_t limit_w1_low = -1;
    int32_t limit_w1_high = -1;
    int32_t limit_w2_low = -1;
    int32_t limit_w2_high = -1;


    int32_t get_size() const{
        return range_size;
    }

    const int32_t* get_range() const{
        return range;
    }

    IDSelectorIVFClusterAwareIntersectDirect(const int32_t* in_cluster_positions, const int32_t* limits, const int16_t* clusters, const int32_t* cluster_limits);

    void set_words(int32_t w1, int32_t w2=-1);

    // initialize the selector for the specified list
    void set_list(idx_t list_no) const override;

    bool is_member(idx_t id) const override;

    ~IDSelectorIVFClusterAwareIntersectDirect(
            ) override {
        delete(tmp);
    };
};




} // namespace faiss
