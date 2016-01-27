/*
 * MyMapper.hh
 *
 *  Created on: Jan 9, 2015
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef MYMAPPER_HH_
#define MYMAPPER_HH_

#include <set>
#include <vector>

#include "legion_types.h"
#include "default_mapper.h"


class MyMapper : public LegionRuntime::HighLevel::DefaultMapper {
public:
  MyMapper(
        LegionRuntime::HighLevel::Machine machine,
        LegionRuntime::HighLevel::HighLevelRuntime *rt,
        LegionRuntime::HighLevel::Processor local);
public:
  virtual void select_task_options(LegionRuntime::HighLevel::Task *task);
  virtual bool map_task(LegionRuntime::HighLevel::Task *task);
  virtual bool rank_copy_targets(
        const LegionRuntime::HighLevel::Mappable *mappable,
        LegionRuntime::HighLevel::LogicalRegion rebuild_region,
        const std::set<LegionRuntime::HighLevel::Memory> &current_instances,
        bool complete,
        size_t max_blocking_factor,
        std::set<LegionRuntime::HighLevel::Memory> &to_reuse,
        std::vector<LegionRuntime::HighLevel::Memory> &to_create,
        bool &create_one,
        size_t &blocking_factor);
};


#endif /* MYMAPPER_HH_ */
