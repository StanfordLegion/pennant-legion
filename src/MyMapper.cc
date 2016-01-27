/*
 * MyMapper.cc
 *
 *  Created on: Jan 9, 2015
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "MyMapper.hh"

#include <cstdlib>
#include <string>
#include <iostream>

#include "legion.h"
#include "default_mapper.h"

using namespace std;
using namespace LegionRuntime::HighLevel;


MyMapper::MyMapper(
        Machine m,
        HighLevelRuntime *rt,
        Processor p)
        : DefaultMapper(m, rt, p) {}


bool MyMapper::map_task(Task *task)
{
    DefaultMapper::map_task(task);
    for (unsigned idx = 0; idx < task->regions.size(); idx++)
    {
        // use max blocking factor => SOA
        task->regions[idx].blocking_factor = task->regions[idx].max_blocking_factor;
    }
    // Report successful mapping results
    return true;
}

void MyMapper::select_task_options(LegionRuntime::HighLevel::Task *task) {
  task->map_locally = true;
  return; 
}
bool MyMapper::rank_copy_targets(
        const Mappable *mappable,
        LogicalRegion rebuild_region,
        const set<Memory> &current_instances,
        bool complete,
        size_t max_blocking_factor,
        set<Memory> &to_reuse,
        vector<Memory> &to_create,
        bool &create_one,
        size_t &blocking_factor)
{
    DefaultMapper::rank_copy_targets(mappable,
            rebuild_region,
            current_instances,
            complete,
            max_blocking_factor,
            to_reuse,
            to_create,
            create_one, blocking_factor);
    // use max blocking factor => SOA
    blocking_factor = max_blocking_factor;
    // return true => use composite instances
#ifdef SHARED_LOWLEVEL
    return false;
#else
    return true;
#endif
}

