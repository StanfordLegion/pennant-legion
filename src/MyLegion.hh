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
#include "Vec2.hh"
#include <cassert>

#ifndef OMP_CHUNK_SIZE
#define OMP_CHUNK_SIZE    32
#endif

#ifndef OMP_SCHEDULE
#define OMP_SCHEDULE      dynamic
#endif

typedef Legion::Point<1,Legion::coord_t> Pointer;

class PointIterator : public Legion::PointInDomainIterator<1,Legion::coord_t> {
public:
  inline PointIterator(Legion::Runtime *rt, Legion::IndexSpace is)
    : Legion::PointInDomainIterator<1,Legion::coord_t>(
        rt->get_index_space_domain(Legion::IndexSpaceT<1,Legion::coord_t>(is))) { }
};

#ifdef NAN_CHECK
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
template<typename ACC>
inline void check_double_nan(const ACC &acc, const Legion::PhysicalRegion &region)
{
  Legion::LogicalRegion handle = region.get_logical_region();  
  for (PointIterator itr(Legion::Runtime::get_runtime(), 
                          handle.get_index_space()); itr(); itr++)
  {
    const double value = acc[*itr];
    // NaNs do not compare equal to themselves
    assert(value == value);
  }
}

template<typename ACC>
inline void check_double2_nan(const ACC &acc, const Legion::PhysicalRegion &region)
{
  Legion::LogicalRegion handle = region.get_logical_region();  
  for (PointIterator itr(Legion::Runtime::get_runtime(), 
                          handle.get_index_space()); itr(); itr++)
  {
    const double2 value = acc[*itr];
    // NaNs do not compare equal to themselves
    assert(value.x == value.x);
    assert(value.y == value.y);
  }
}

// We provide specialized accessors here for double and double2 that check
// for NaN values on creation for read privileges and on destruction for
// write privileges
template<typename T>
class AccessorRO : public Legion::FieldAccessor<READ_ONLY,T,1,Legion::coord_t,
                                Realm::AffineAccessor<T,1,Legion::coord_t> >
{
public:
  AccessorRO(const Legion::PhysicalRegion &region, Legion::FieldID fid)
    : Legion::FieldAccessor<READ_ONLY,T,1,Legion::coord_t,
        Realm::AffineAccessor<T,1,Legion::coord_t> >(region, fid) { }
};

// specialization for double
template<>
class AccessorRO<double> : public Legion::FieldAccessor<READ_ONLY,double,1,Legion::coord_t,
                                Realm::AffineAccessor<double,1,Legion::coord_t> >
{
public:
  AccessorRO(const Legion::PhysicalRegion &region, Legion::FieldID fid)
    : Legion::FieldAccessor<READ_ONLY,double,1,Legion::coord_t,
        Realm::AffineAccessor<double,1,Legion::coord_t> >(region, fid) 
  { check_double_nan(*this, region); } 
};

// specialization for double2
template<>
class AccessorRO<double2> : public Legion::FieldAccessor<READ_ONLY,double2,1,Legion::coord_t,
                                Realm::AffineAccessor<double2,1,Legion::coord_t> >
{
public:
  AccessorRO(const Legion::PhysicalRegion &region, Legion::FieldID fid)
    : Legion::FieldAccessor<READ_ONLY,double2,1,Legion::coord_t,
        Realm::AffineAccessor<double2,1,Legion::coord_t> >(region, fid)
  { check_double2_nan(*this, region); }
};

template<typename T>
class AccessorWD : public Legion::FieldAccessor<WRITE_DISCARD,T,1,Legion::coord_t,
                                Realm::AffineAccessor<T,1,Legion::coord_t> >
{
public:
  AccessorWD(const Legion::PhysicalRegion &reg, Legion::FieldID fid)
    : Legion::FieldAccessor<WRITE_DISCARD,T,1,Legion::coord_t,
        Realm::AffineAccessor<T,1,Legion::coord_t> >(reg, fid)
  { }
};

// specialization for double
template<>
class AccessorWD<double> : 
            public Legion::FieldAccessor<WRITE_DISCARD,double,1,Legion::coord_t,
                                Realm::AffineAccessor<double,1,Legion::coord_t> >
{
public:
  AccessorWD(const Legion::PhysicalRegion &reg, Legion::FieldID fid)
    : Legion::FieldAccessor<WRITE_DISCARD,double,1,Legion::coord_t,
        Realm::AffineAccessor<double,1,Legion::coord_t> >(reg, fid), 
        region(reg), field(fid)
  { }
  ~AccessorWD(void)
#ifdef __CUDACC__
  { cudaDeviceSynchronize(); check_double_nan(*this, region); }
#else
  { check_double_nan(*this, region); }
#endif
public:
  const Legion::PhysicalRegion &region;
  const Legion::FieldID field;
};

// specialization for double2
template<>
class AccessorWD<double2> : 
            public Legion::FieldAccessor<WRITE_DISCARD,double2,1,Legion::coord_t,
                                Realm::AffineAccessor<double2,1,Legion::coord_t> >
{
public:
  AccessorWD(const Legion::PhysicalRegion &reg, Legion::FieldID fid)
    : Legion::FieldAccessor<WRITE_DISCARD,double2,1,Legion::coord_t,
        Realm::AffineAccessor<double2,1,Legion::coord_t> >(reg, fid), 
        region(reg), field(fid)
  { }
  ~AccessorWD(void)
#ifdef __CUDACC__
  { cudaDeviceSynchronize(); check_double2_nan(*this, region); }
#else
  { check_double2_nan(*this, region); }
#endif
public:
  const Legion::PhysicalRegion &region;
  const Legion::FieldID field;
};

template<typename T>
class AccessorRW : public Legion::FieldAccessor<READ_WRITE,T,1,Legion::coord_t,
                                Realm::AffineAccessor<T,1,Legion::coord_t> >
{
public:
  AccessorRW(const Legion::PhysicalRegion &reg, Legion::FieldID fid)
    : Legion::FieldAccessor<READ_WRITE,T,1,Legion::coord_t,
        Realm::AffineAccessor<T,1,Legion::coord_t> >(reg, fid) { }
};

// specialization for double
template<>
class AccessorRW<double> : public Legion::FieldAccessor<READ_WRITE,double,1,Legion::coord_t,
                                Realm::AffineAccessor<double,1,Legion::coord_t> >
{
public:
  AccessorRW(const Legion::PhysicalRegion &reg, Legion::FieldID fid)
    : Legion::FieldAccessor<READ_WRITE,double,1,Legion::coord_t,
        Realm::AffineAccessor<double,1,Legion::coord_t> >(reg, fid), 
        region(reg), field(fid)
  { check_double_nan(*this, region); }
  ~AccessorRW(void)
#ifdef __CUDACC__
  { cudaDeviceSynchronize(); check_double_nan(*this, region); }
#else
  { check_double_nan(*this, region); }
#endif
public:
  const Legion::PhysicalRegion &region;
  const Legion::FieldID field;
};

// specialization for double2
template<>
class AccessorRW<double2> : public Legion::FieldAccessor<READ_WRITE,double2,1,Legion::coord_t,
                                Realm::AffineAccessor<double2,1,Legion::coord_t> >
{
public:
  AccessorRW(const Legion::PhysicalRegion &reg, Legion::FieldID fid)
    : Legion::FieldAccessor<READ_WRITE,double2,1,Legion::coord_t,
        Realm::AffineAccessor<double2,1,Legion::coord_t> >(reg, fid), 
        region(reg), field(fid)
  { check_double2_nan(*this, region); }
  ~AccessorRW(void)
#ifdef __CUDACC__
  { cudaDeviceSynchronize(); check_double2_nan(*this, region); }
#else
  { check_double2_nan(*this, region); }
#endif
public:
  const Legion::PhysicalRegion &region;
  const Legion::FieldID field;
};
#else
template <typename T>
using AccessorRO = typename Legion::FieldAccessor<READ_ONLY,T,1,Legion::coord_t,
                                Realm::AffineAccessor<T,1,Legion::coord_t> >;
template <typename T>
using AccessorWD = typename Legion::FieldAccessor<WRITE_DISCARD,T,1,Legion::coord_t,
                                Realm::AffineAccessor<T,1,Legion::coord_t> >;
template <typename T>
using AccessorRW = typename Legion::FieldAccessor<READ_WRITE,T,1,Legion::coord_t,
                                Realm::AffineAccessor<T,1,Legion::coord_t> >;
#endif

template <typename REDOP, bool EXCLUSIVE=true>
using AccessorRD = typename Legion::ReductionAccessor<REDOP,EXCLUSIVE,1,
      Legion::coord_t, Realm::AffineAccessor<typename REDOP::RHS,1,Legion::coord_t> >;

#endif /* MYLEGION_HH_ */
