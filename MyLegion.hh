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


// // convenience class for accessors
// // (this would be a template typedef if C++98 allowed it)
// template <typename T>
// struct MyAccessor : public LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::SOA<sizeof(T)>, T> {
//     typedef LegionRuntime::Accessor::AccessorType::SOA<sizeof(T)> accessor_type_t;
//     typedef LegionRuntime::Accessor::RegionAccessor<accessor_type_t, T> accessor_t;

//     MyAccessor(const accessor_t& rhs) : accessor_t(rhs) {}

// };



template <typename T>
using MyAccessorType = typename LegionRuntime::Accessor::AccessorType::SOA<sizeof(T)>;

template <typename T>
  using MyAccessor = typename LegionRuntime::Accessor::RegionAccessor<MyAccessorType<T> , T>;

// convenience function for getting accessors
template <typename T>
MyAccessor<T> get_accessor(
        const LegionRuntime::HighLevel::PhysicalRegion& region,
        const LegionRuntime::HighLevel::FieldID fid) {
  //typename MyAccessor<T>::accessor_t my_accessor =
  return region.get_field_accessor(fid).typeify<T>().
    template convert<MyAccessorType<T>>();
  //std::cout << &my_accessor << std::endl;
  //return my_accessor; 
}


// convenience function for getting accessors
// template <typename T>
// typename MyAccessor<T>::accessor_t get_accessor(
//         const LegionRuntime::HighLevel::PhysicalRegion& region,
//         const LegionRuntime::HighLevel::FieldID fid) {
//   //typename MyAccessor<T>::accessor_t my_accessor =
//   return region.get_field_accessor(fid).typeify<T>().
//     template convert<typename MyAccessor<T>::accessor_type_t>();
//   //std::cout << &my_accessor << std::endl;
//   //return my_accessor; 
// }


// convenience class for reduction accessors
// (this would be a template typedef if C++98 allowed it)
template <typename OP>
struct MyReductionAccessor : public LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::ReductionFold<OP>, typename OP::RHS> {
    typedef LegionRuntime::Accessor::AccessorType::ReductionFold<OP> accessor_type_t;
    typedef LegionRuntime::Accessor::RegionAccessor<accessor_type_t, typename OP::RHS> accessor_t;

    MyReductionAccessor(const accessor_t& rhs) : accessor_t(rhs) {}

};


// convenience function for getting reduction accessors
template <typename OP>
typename MyReductionAccessor<OP>::accessor_t get_reduction_accessor(
        const LegionRuntime::HighLevel::PhysicalRegion& region) {
    return region.get_accessor().typeify<typename OP::RHS>().
        template convert<typename MyReductionAccessor<OP>::accessor_type_t>();
}


#endif /* MYLEGION_HH_ */
