/*
 * MyLegion.hh
 *
 *  Created on: Nov 26, 2014
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef MYLEGION_HH_
#define MYLEGION_HH_

#include "legion.h"

template <typename T>
using AccessorRO = typename Legion::FieldAccessor<READ_ONLY,T,1,Legion::coord_t,
                                Realm::AffineAccessor<T,1,Legion::coord_t> >;
template <typename T>
using AccessorWD = typename Legion::FieldAccessor<WRITE_DISCARD,T,1,Legion::coord_t,
                                Realm::AffineAccessor<T,1,Legion::coord_t> >;
template <typename T>
using AccessorRW = typename Legion::FieldAccessor<READ_WRITE,T,1,Legion::coord_t,
                                Realm::AffineAccessor<T,1,Legion::coord_t> >;
template <typename REDOP>
using AccessorRD = typename Legion::ReductionAccessor<REDOP,false/*exclusive*/,1,
      Legion::coord_t, Realm::AffineAccessor<typename REDOP::RHS,1,Legion::coord_t> >;

typedef Legion::Point<1,Legion::coord_t> Pointer;

class PointIterator : public Legion::PointInDomainIterator<1,Legion::coord_t> {
public:
  inline PointIterator(Legion::Runtime *rt, Legion::IndexSpace is)
    : Legion::PointInDomainIterator<1,Legion::coord_t>(
        rt->get_index_space_domain(Legion::IndexSpaceT<1,Legion::coord_t>(is))) { }
};

#endif /* MYLEGION_HH_ */
