/*
 * PennantMapper.cc
 *
 *  Created on: Jan 9, 2015
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "Mesh.hh"
#include "PennantMapper.hh"

#include <cstdlib>
#include <string>
#include <iostream>

#include "legion.h"
#include "default_mapper.h"

using namespace std;
using namespace Legion;
using namespace Legion::Mapping;

PennantMapper::PennantMapper(
        Machine m,
        Runtime *rt,
        Processor p)
  : DefaultMapper(rt->get_mapper_runtime(), m, p), 
    pennant_mapper_name(get_name(p)), numpcx(0), numpcy(0), sharded(false)
{
  // Get our local memories
  {
    Machine::MemoryQuery sysmem_query(machine);
    sysmem_query.local_address_space();
    sysmem_query.only_kind(Memory::SYSTEM_MEM);
    local_sysmem = sysmem_query.first();
    assert(local_sysmem.exists());
  }
  if (!local_gpus.empty()) {
    Machine::MemoryQuery zc_query(machine);
    zc_query.local_address_space();
    zc_query.only_kind(Memory::Z_COPY_MEM);
    local_zerocopy = zc_query.first();
    assert(local_zerocopy.exists());
  } else {
    local_zerocopy = Memory::NO_MEMORY;
  }
  if (local_kind == Processor::TOC_PROC) {
    Machine::MemoryQuery fb_query(machine);
    fb_query.local_address_space();
    fb_query.only_kind(Memory::GPU_FB_MEM);
    fb_query.best_affinity_to(local_proc);
    local_framebuffer = fb_query.first();
    assert(local_framebuffer.exists());
  } else {
    local_framebuffer = Memory::NO_MEMORY;
  }
  if (local_kind == Processor::OMP_PROC) {
    Machine::MemoryQuery numa_query(machine);
    numa_query.local_address_space();
    numa_query.only_kind(Memory::SOCKET_MEM);
    numa_query.best_affinity_to(local_proc);
    local_numa = numa_query.first();
    assert(local_numa.exists());
  } else {
    local_numa = Memory::NO_MEMORY;
  }
}

PennantMapper::~PennantMapper(void)
{
  free(const_cast<char*>(pennant_mapper_name));
}

const char* PennantMapper::get_mapper_name(void) const
{
  return pennant_mapper_name;
}

Mapper::MapperSyncModel PennantMapper::get_mapper_sync_model(void) const
{
  return SERIALIZED_REENTRANT_MAPPER_MODEL;
}

bool PennantMapper::request_valid_instances(void) const
{
  // For now we don't need this information
  return false;
}

void PennantMapper::select_tunable_value(const MapperContext ctx,
                                         const Task& task,
                                         const SelectTunableInput& input,
                                               SelectTunableOutput& output)
{
  // No custom penant tunable values yet
  DefaultMapper::select_tunable_value(ctx, task, input, output);
}

void PennantMapper::select_task_options(const MapperContext ctx,
                                        const Task& task,
                                              TaskOptions& output)
{
  // The default mapper mostly does the right thing
  DefaultMapper::select_task_options(ctx, task, output);
  // But we don't need the valid instances
  output.valid_instances = false;
#ifdef PENNANT_DISABLE_CONTROL_REPLICATION
  output.replicate = false;
  output.map_locally = false;
#endif
}

Processor PennantMapper::default_policy_select_initial_processor(
                                    MapperContext ctx, const Task &task)
{
  // Always keep it on our local processor
  // Index tasks will get distributed by sharding, single tasks will stay local
  return task.current_proc;
}

void PennantMapper::slice_task(const MapperContext ctx,
                               const Task &task,
                               const SliceTaskInput &input,
                                     SliceTaskOutput &output)
{
#ifdef PENNANT_DISABLE_CONTROL_REPLICATION
  // If this is the first slice_task call with no control replication
  // then we need to emulate the effect of the sharding here
  if ((task.index_domain == input.domain) && (total_nodes > 1)) {
    // Do this computation so we can get per shard rectangles on the remote node
    if (!sharded)
      compute_fake_sharding(ctx);
    output.slices.resize(sharding_spaces.size());
    unsigned index = 0;
    for (std::vector<std::pair<Processor,IndexSpace> >::const_iterator it = 
          sharding_spaces.begin(); it != sharding_spaces.end(); it++, index++)
      output.slices[index] = 
        TaskSlice(it->second, it->first, true/*recurse*/, false/*stealable*/);
    return;
  }
#endif
  // We've already been control replicated, so just divide our points
  // over the local processors, depending on which kind we prefer
  if ((task.tag & PREFER_GPU) && !local_gpus.empty()) {
    unsigned local_gpu_index = 0;
    for (Domain::DomainPointIterator itr(input.domain); itr; itr++)
    {
      TaskSlice slice;
      slice.domain = Domain(itr.p, itr.p);
      slice.proc = local_gpus[local_gpu_index++];
      if (local_gpu_index == local_gpus.size())
        local_gpu_index = 0;
      slice.recurse = false;
      slice.stealable = false;
      output.slices.push_back(slice);
    }
  } else if ((task.tag & PREFER_OMP) && !local_omps.empty()) {
    unsigned local_omp_index = 0;
    for (Domain::DomainPointIterator itr(input.domain); itr; itr++)
    {
      TaskSlice slice;
      slice.domain = Domain(itr.p, itr.p);
      slice.proc = local_omps[local_omp_index++];
      if (local_omp_index == local_omps.size())
        local_omp_index = 0;
      slice.recurse = false;
      slice.stealable = false;
      output.slices.push_back(slice);
    }
  } else {
    // Opt for our cpus instead of our openmap processors
    unsigned local_cpu_index = 0;
    for (Domain::DomainPointIterator itr(input.domain); itr; itr++)
    {
      TaskSlice slice;
      slice.domain = Domain(itr.p, itr.p);
      slice.proc = local_cpus[local_cpu_index++];
      if (local_cpu_index == local_cpus.size())
        local_cpu_index = 0;
      slice.recurse = false;
      slice.stealable = false;
      output.slices.push_back(slice);
    }
  }
}

void PennantMapper::map_task(const MapperContext ctx,
                             const Task &task,
                             const MapTaskInput &input,
                                   MapTaskOutput &output)
{
  if ((task.tag & PREFER_GPU) && !local_gpus.empty()) {
    output.chosen_variant = find_gpu_variant(ctx, task.task_id);
    output.target_procs.push_back(task.target_proc);
  } else if ((task.tag & PREFER_OMP) && !local_omps.empty()) {
    output.chosen_variant = find_omp_variant(ctx, task.task_id);
    output.target_procs.push_back(task.target_proc);
  } else {
    output.chosen_variant = find_cpu_variant(ctx, task.task_id);
    output.target_procs = local_cpus;
  }
  output.chosen_instances.resize(task.regions.size());  
  if ((task.tag & PREFER_GPU) && !local_gpus.empty()) {
    for (unsigned idx = 0; idx < task.regions.size(); idx++)
    {
      // See if it is a reduction region requirement or not
      if (task.regions[idx].privilege == REDUCE)
        create_reduction_instances(ctx, task, idx, 
#ifdef NAN_CHECK
        // If we're doing nan-checks make sure things are in zero-copy
        // so we can read-the values directly from the host
                                   local_zerocopy,
#else
                                   local_framebuffer,
#endif
                                   output.chosen_instances[idx]);
      else if (task.regions[idx].tag & PREFER_ZCOPY)
        map_pennant_array(ctx, task, idx, task.regions[idx].region,
                          local_zerocopy,
                          output.chosen_instances[idx]);
      else
        map_pennant_array(ctx, task, idx, task.regions[idx].region, 
#ifdef NAN_CHECK
        // If we're doing nan-checks make sure things are in zero-copy
        // so we can read-the values directly from the host
                          local_zerocopy,
#else
                          local_framebuffer,
#endif
                          output.chosen_instances[idx]);
    }
  } else {
    for (unsigned idx = 0; idx < task.regions.size(); idx++)
    {
      // See if it is a reduction region requirement or not
      if (task.regions[idx].privilege == REDUCE)
        create_reduction_instances(ctx, task, idx, 
            local_numa.exists() ? local_numa : local_sysmem,
                                   output.chosen_instances[idx]);
      else
        map_pennant_array(ctx, task, idx, task.regions[idx].region, 
            local_numa.exists() ? local_numa : local_sysmem, 
                          output.chosen_instances[idx]);
    }
  }
  runtime->acquire_instances(ctx, output.chosen_instances);
  // Finally set the priority for the task
  if (task.tag & CRITICAL)
    output.task_priority = 1;
  else
    output.task_priority = 0;
}

void PennantMapper::speculate(const MapperContext ctx,
                              const Task &task,
                                    SpeculativeOutput &output)
{
#ifdef ENABLE_MAX_CYCLE_PREDICATION
  output.speculate = true;
  output.speculative_value = true; // not done 
  output.speculate_mapping_only = true;
#else
  output.speculate = false;
#endif
}

#ifndef NO_LEGION_CONTROL_REPLICATION
void PennantMapper::select_sharding_functor(const MapperContext ctx,
                                            const Task &task,
                                            const SelectShardingFunctorInput &input,
                                                  SelectShardingFunctorOutput &output)
{
  if (task.is_index_space)
    output.chosen_functor = PENNANT_SHARD_ID; 
  else
    output.chosen_functor = 0; // default sharding functor so everything is on node 0
}
#endif

void PennantMapper::map_copy(const MapperContext ctx,
                             const Copy &copy,
                             const MapCopyInput &input,
                                   MapCopyOutput &output)
{
  output.src_instances.resize(copy.src_requirements.size());
  output.dst_instances.resize(copy.dst_requirements.size());
  output.src_indirect_instances.resize(copy.src_indirect_requirements.size());
  // There should be no scatter copies
  assert(copy.dst_indirect_requirements.empty());
  // Keep the gather copies on the host side
  if (!local_gpus.empty() && copy.src_indirect_requirements.empty()) {
    assert(copy.is_index_space);
    const Point<1> point = copy.index_point;
#ifdef PENNANT_DISABLE_CONTROL_REPLICATION
    if (!sharded)
      compute_fake_sharding(ctx);
    const Memory fbmem = sharding_memories[point];
#else
    const coord_t index = compute_shard_index(point);
    const Processor gpu = local_gpus[index % local_gpus.size()];
    const Memory fbmem = default_policy_select_target_memory(ctx, gpu,
                                        copy.src_requirements.front());
#endif
    assert(fbmem.kind() == Memory::GPU_FB_MEM);
    for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++)
      map_pennant_array(ctx, copy, idx, copy.src_requirements[idx].region, fbmem,
                        output.src_instances[idx]);
    for (unsigned idx = 0; idx < copy.dst_requirements.size(); idx++)
      map_pennant_array(ctx, copy, idx + copy.src_requirements.size(), 
                        copy.dst_requirements[idx].region, fbmem,
                        output.dst_instances[idx]);
  } else if (!local_omps.empty() && copy.src_indirect_requirements.empty()) {
    assert(copy.is_index_space);
    const Point<1> point = copy.index_point;
#ifdef PENNANT_DISABLE_CONTROL_REPLICATION
    if (!sharded)
      compute_fake_sharding(ctx);
    const Memory numa = sharding_memories[point];
#else
    const coord_t index = compute_shard_index(point);
    const Processor omp = local_omps[index % local_omps.size()];
    const Memory numa = default_policy_select_target_memory(ctx, omp,
                                        copy.src_requirements.front());
#endif
    assert(numa.kind() == Memory::SOCKET_MEM);
    for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++)
      map_pennant_array(ctx, copy, idx, copy.src_requirements[idx].region, numa,
                        output.src_instances[idx]);
    for (unsigned idx = 0; idx < copy.dst_requirements.size(); idx++)
      map_pennant_array(ctx, copy, idx + copy.src_requirements.size(), 
                        copy.dst_requirements[idx].region, numa,
                        output.dst_instances[idx]);
  } else {
#ifdef PENNANT_DISABLE_CONTROL_REPLICATION
    assert(copy.is_index_space);
    const Point<1> point = copy.index_point;
    if (!sharded)
      compute_fake_sharding(ctx);
    const Memory sysmem = sharding_sys_memories[point];
#else
    const Memory sysmem = local_sysmem;
#endif
    for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++)
      map_pennant_array(ctx, copy, idx, copy.src_requirements[idx].region, sysmem,
                        output.src_instances[idx]);
    for (unsigned idx = 0; idx < copy.dst_requirements.size(); idx++)
      map_pennant_array(ctx, copy, idx + copy.src_requirements.size(), 
                        copy.dst_requirements[idx].region, sysmem,
                        output.dst_instances[idx]);
    if (!copy.src_indirect_requirements.empty())
    {
      const size_t offset = copy.src_requirements.size() + copy.dst_requirements.size();
      for (unsigned idx = 0; idx < copy.src_indirect_requirements.size(); idx++)
      {
        std::vector<PhysicalInstance> insts;
        map_pennant_array(ctx, copy, idx + offset, 
                          copy.src_indirect_requirements[idx].region, sysmem, insts);
        assert(insts.size() == 1);
        output.src_indirect_instances[idx] = insts[0];
      }
    }
  }
  runtime->acquire_instances(ctx, output.src_instances);
  runtime->acquire_instances(ctx, output.dst_instances);
}

void PennantMapper::speculate(const MapperContext ctx,
                              const Copy &copy,
                                    SpeculativeOutput &output)
{
#ifdef ENABLE_MAX_CYCLE_PREDICATION
  output.speculate = true;
  output.speculative_value = true; // not done 
  output.speculate_mapping_only = true;
#else
  output.speculate = false;
#endif
}

#ifndef NO_LEGION_CONTROL_REPLICATION
void PennantMapper::select_sharding_functor(const MapperContext ctx,
                                            const Copy &copy,
                                            const SelectShardingFunctorInput &input,
                                                  SelectShardingFunctorOutput &output)
{
  output.chosen_functor = PENNANT_SHARD_ID; 
}
#endif

void PennantMapper::select_partition_projection(const MapperContext  ctx,
                                                const Partition& partition,
                                                const SelectPartitionProjectionInput& input,
                                                      SelectPartitionProjectionOutput& output)
{
  if (!input.open_complete_partitions.empty()) {
#ifdef PENNANT_DISABLE_CONTROL_REPLICATION
    const Domain color_space = 
      runtime->get_index_partition_color_space(ctx, 
          input.open_complete_partitions.front().get_index_partition());
    if (color_space.get_volume() != sharding_memories.size())
      return;
#endif
    output.chosen_partition = input.open_complete_partitions.front();
    return;
  }
  // Otherwise see if we can find another complete partition for partition
  for (Color color = 0; color < 4; color++)
  {
    if (!runtime->has_index_partition(ctx, 
          partition.requirement.region.get_index_space(), color))
      continue;
    IndexPartition ip = runtime->get_index_partition(ctx, 
        partition.requirement.region.get_index_space(), color);
    if (!runtime->is_index_partition_complete(ctx, ip))
      continue;
#ifdef PENNANT_DISABLE_CONTROL_REPLICATION
    const Domain color_space = runtime->get_index_partition_color_space(ctx, ip);
    if (color_space.get_volume() != sharding_memories.size())
      continue;
#endif
    output.chosen_partition = runtime->get_logical_partition(ctx, 
        partition.requirement.region, ip);
    return;
  }
}

void PennantMapper::map_partition(const MapperContext ctx,
                                  const Partition& partition,
                                  const MapPartitionInput&   input,
                                        MapPartitionOutput&  output)
{
#ifdef PENNANT_DISABLE_CONTROL_REPLICATION
  Memory sysmem;
  if (partition.is_index_space) {
    assert(partition.is_index_space);
    const Point<1> point = partition.index_point;
    if (!sharded)
      compute_fake_sharding(ctx);
    sysmem = sharding_sys_memories[point];
  } else {
    sysmem = local_sysmem;
  }
#else
  const Memory sysmem = local_sysmem;
#endif
  map_pennant_array(ctx, partition, 0, partition.requirement.region, sysmem,
                    output.chosen_instances);
  runtime->acquire_instances(ctx, output.chosen_instances);
}

void PennantMapper::memoize_operation(const MapperContext  ctx,
                                      const Mappable&      mappable,
                                      const MemoizeInput&  input,
                                            MemoizeOutput& output)
{
#ifdef ENABLE_MAX_CYCLE_PREDICATION
  // Legion doesn't support tracing with predication yet
  output.memoize = false;
#else
  output.memoize = true;
#endif
}

/*static*/ const char* PennantMapper::get_name(Processor p)
{
  char *result = (char*)malloc(256);
  snprintf(result, 256, "Pennant Mapper on Processor %llx", p.id);
  return result;
}

#ifndef NO_LEGION_CONTROL_REPLICATION
void PennantMapper::select_sharding_functor(const MapperContext ctx,
                                            const Partition &partition,
                                            const SelectShardingFunctorInput &input,
                                                  SelectShardingFunctorOutput &output)
{
  output.chosen_functor = PENNANT_SHARD_ID; 
}

void PennantMapper::select_sharding_functor(const MapperContext ctx,
                                            const Fill &fill,
                                            const SelectShardingFunctorInput &input,
                                                  SelectShardingFunctorOutput &output)
{
  output.chosen_functor = PENNANT_SHARD_ID; 
}
#endif

void PennantMapper::map_pennant_array(const MapperContext ctx,
                                      const Mappable &mappable, unsigned index, 
                                      LogicalRegion region, Memory target,
                                      std::vector<PhysicalInstance> &instances,
                                      bool initialization_instance)
{
  const std::pair<LogicalRegion,Memory> key(region, target);
  std::map<std::pair<LogicalRegion,Memory>,PhysicalInstance>::const_iterator
    finder = local_instances.find(key);
  if (finder != local_instances.end()) {
    instances.push_back(finder->second);
    return;
  }
  // First time through make an instance

  // Make a big instance of the top-level region for all
  // single-node CPU runs and any single-node single-GPU runs
  if ((total_nodes == 1) && 
      ((target.kind() != Memory::GPU_FB_MEM) || (local_gpus.size() == 1)))
        
  {
    while (runtime->has_parent_index_partition(ctx, region.get_index_space())) 
    {
      LogicalPartition part = runtime->get_parent_logical_partition(ctx, region);
      region = runtime->get_parent_logical_region(ctx, part);
    }
  }
  std::vector<LogicalRegion> regions(1, region);
  LayoutConstraintSet layout_constraints;
  // No specialization
  layout_constraints.add_constraint(SpecializedConstraint());
  // SOA dimension ordering (all pennant arrays are 1-D)
  std::vector<DimensionKind> dimension_ordering(2);
  dimension_ordering[0] = DIM_X;
  dimension_ordering[1] = DIM_F;
  layout_constraints.add_constraint(OrderingConstraint(dimension_ordering,
                                                       false/*contiguous*/));
  // Constrained for the target memory kind
  layout_constraints.add_constraint(MemoryConstraint(target.kind()));
  // Have all the fields for the instance available
  std::vector<FieldID> all_fields;
  runtime->get_field_space_fields(ctx, region.get_field_space(), all_fields);
  layout_constraints.add_constraint(
      FieldConstraint(all_fields, false/*contiguous*/, false/*inorder*/));
  PhysicalInstance result; bool created;
  if (!runtime->find_or_create_physical_instance(ctx, target, layout_constraints,
        regions, result, created, true/*acquire*/, 
        initialization_instance ? 0/*normal GC priority*/ : GC_NEVER_PRIORITY)) {
    fprintf(stderr,"Pennant mapper is out of memory!\n");
    switch (mappable.get_mappable_type())
    {
      case Mappable::TASK_MAPPABLE:
        {
          const Task *task = mappable.as_task();
          fprintf(stderr,"ERROR: Pennant mapper %s failed to allocate instance in "
                  "memory " IDFMT " of kind %d for region requirement %d of task %s!\n",
                  get_mapper_name(), target.id, target.kind(), index, task->get_task_name());
          break;
        }
      case Mappable::COPY_MAPPABLE:
        {
          fprintf(stderr,"ERROR: Pennant mapper %s failed to allocate instance in "
                  "memory " IDFMT " of kind %d for region requirement %d of copy!\n",
                  get_mapper_name(), target.id, target.kind(), index);
          break;
        }
      case Mappable::PARTITION_MAPPABLE:
        {
          fprintf(stderr,"ERROR: Pennant mapper %s failed to allocate instance in "
                  "memory " IDFMT " of kind %d for region requirement %d of partition!\n",
                  get_mapper_name(), target.id, target.kind(), index);
          break;
        }
      default:
        fprintf(stderr,"ERROR: Pennant mapper %s failed to allocate instance in "
                "memory " IDFMT " of kind %d for region requirement %d of unknown mappable!\n",
                get_mapper_name(), target.id, target.kind(), index);
    }
    assert(false);
  }
  instances.push_back(result);
  // Save the result for future use
  local_instances[key] = result;
}

void PennantMapper::create_reduction_instances(const MapperContext ctx,
                                               const Task &task, unsigned index,
                                               Memory target_memory,
                                               std::vector<PhysicalInstance> &instances)
{
  std::set<FieldID> dummy_fields;
  TaskLayoutConstraintSet dummy_constraints;
  if (!default_create_custom_instances(ctx, task.target_proc, target_memory,
      task.regions[index], index, dummy_fields,
      dummy_constraints, false/*need check*/, instances)) {
    fprintf(stderr,"Pennant mapper is out of memory!\n");
    fprintf(stderr,"ERROR: Pennant mapper %s failed to allocate reduction instance "
            "in memory " IDFMT " of kind %d for region requirement %d of task %s!\n",
            get_mapper_name(), target_memory.id, target_memory.kind(), 
            index, task.get_task_name());
    assert(false);
  }
}

VariantID PennantMapper::find_cpu_variant(const MapperContext ctx, TaskID task_id)
{
  std::map<TaskID,VariantID>::const_iterator finder = 
    cpu_variants.find(task_id);
  if (finder != cpu_variants.end())
    return finder->second;
  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, task_id, variants, Processor::LOC_PROC);
  assert(variants.size() == 1); // should be exactly one for pennant 
  cpu_variants[task_id] = variants[0];
  return variants[0];
}

VariantID PennantMapper::find_omp_variant(const MapperContext ctx, TaskID task_id)
{
  std::map<TaskID,VariantID>::const_iterator finder = 
    omp_variants.find(task_id);
  if (finder != omp_variants.end())
    return finder->second;
  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, task_id, variants, Processor::OMP_PROC);
  assert(variants.size() == 1); // should be exactly one for pennant 
  omp_variants[task_id] = variants[0];
  return variants[0];
}

VariantID PennantMapper::find_gpu_variant(const MapperContext ctx, TaskID task_id)
{
  std::map<TaskID,VariantID>::const_iterator finder = 
    gpu_variants.find(task_id);
  if (finder != gpu_variants.end())
    return finder->second;
  std::vector<VariantID> variants;
  runtime->find_valid_variants(ctx, task_id, variants, Processor::TOC_PROC);
  assert(variants.size() == 1); // should be exactly one for pennant 
  gpu_variants[task_id] = variants[0];
  return variants[0];
}

void PennantMapper::update_mesh_information(coord_t npcx, coord_t npcy)
{
  assert(numpcx == 0);
  assert(numpcy == 0);
  assert(!sharded);
  numpcx = npcx;
  numpcy = npcy;
}

#ifdef PENNANT_DISABLE_CONTROL_REPLICATION
void PennantMapper::compute_fake_sharding(MapperContext ctx)
{
  runtime->disable_reentrant(ctx);
  assert(!sharded);
  assert(numpcx > 0);
  assert(numpcy > 0);
  coord_t nsx, nsy;
  calc_pieces_helper(total_nodes/*number of shards*/, numpcx, numpcy, nsx, nsy);
  // Figure out which shard we are in x and y
  const Point<2> shard_point(node_id % nsx, node_id / nsx);
  const coord_t pershardx = (numpcx + nsx - 1) / nsx;
  const coord_t pershardy = (numpcy + nsy - 1) / nsy;
  const Rect<2> full(Point<2>(0, 0), Point<2>(numpcx-1, numpcy-1));
  sharding_spaces.resize(total_nodes, 
      std::make_pair(Processor::NO_PROC,IndexSpace::NO_SPACE));
  for (std::vector<Processor>::const_iterator rit = 
        remote_cpus.begin(); rit != remote_cpus.end(); rit++) {
    const AddressSpaceID space = rit->address_space();
    if (sharding_spaces[space].first.exists())
      continue;
    sharding_spaces[space].first = *rit;
    const Point<2> shard_point(space % nsx, space / nsx);
    const Rect<2> rect(Point<2>(shard_point[0] * pershardx, shard_point[1] * pershardy),
                       Point<2>((shard_point[0] + 1) * pershardx - 1,
                                (shard_point[1] + 1) * pershardy - 1));
    const Rect<2> space_rect = rect.intersection(full);
    std::vector<Rect<1> > space_rects;
    for (coord_t y = space_rect.lo[1]; y <= space_rect.hi[1]; y++)
      space_rects.push_back(Rect<1>(Point<1>(y * numpcx + space_rect.lo[0]),
                                    Point<1>(y * numpcx + space_rect.hi[0])));
    sharding_spaces[space].second = runtime->create_index_space(ctx, space_rects);
    // compute the sharding memories for all the points on this node
    Machine::MemoryQuery sysmem_query(machine);
    sysmem_query.best_affinity_to(*rit);
    sysmem_query.only_kind(Memory::SYSTEM_MEM);
    assert(sysmem_query.count() > 0);
    const Memory space_sysmem = sysmem_query.first();
    for (coord_t x = space_rect.lo[0]; x <= space_rect.hi[0]; x++) {
      for (coord_t y = space_rect.lo[1]; y <= space_rect.hi[1]; y++) {
        const Point<1> key(y * numpcx + x);
        sharding_sys_memories[key] = space_sysmem;
      }
    }
    if (!local_gpus.empty()) {
      Machine::ProcessorQuery space_gpus(machine);
      space_gpus.same_address_space_as(*rit);
      space_gpus.only_kind(Processor::TOC_PROC);
      assert(space_gpus.count() > 0);
      std::vector<Memory> space_memories;
      for (Machine::ProcessorQuery::iterator it =
            space_gpus.begin(); it != space_gpus.end(); it++)
      {
        Machine::MemoryQuery fbmem_query(machine);
        fbmem_query.best_affinity_to(*it);
        fbmem_query.only_kind(Memory::GPU_FB_MEM);
        assert(fbmem_query.count() > 0);
        space_memories.push_back(fbmem_query.first());
      }
      for (coord_t x = space_rect.lo[0]; x <= space_rect.hi[0]; x++) {
        for (coord_t y = space_rect.lo[1]; y <= space_rect.hi[1]; y++) {
          const Point<1> key(y * numpcx + x);
          const coord_t x2 = x - space_rect.lo[0];
          const coord_t y2 = y - space_rect.lo[1];
          const coord_t index = y2 * ((space_rect.hi[0] - space_rect.lo[0]) + 1) + x2;
          sharding_memories[key] = space_memories[index % space_memories.size()];
        }
      }
    } else if (!local_omps.empty()) {
      Machine::ProcessorQuery space_omps(machine);
      space_omps.same_address_space_as(*rit);
      space_omps.only_kind(Processor::OMP_PROC);
      assert(space_omps.count() > 0);
      std::vector<Memory> space_memories;
      for (Machine::ProcessorQuery::iterator it =
            space_omps.begin(); it != space_omps.end(); it++)
      {
        Machine::MemoryQuery numa_query(machine);
        numa_query.best_affinity_to(*it);
        numa_query.only_kind(Memory::SOCKET_MEM);
        assert(numa_query.count() > 0);
        space_memories.push_back(numa_query.first());
      }
      for (coord_t x = space_rect.lo[0]; x <= space_rect.hi[0]; x++) {
        for (coord_t y = space_rect.lo[1]; y <= space_rect.hi[1]; y++) {
          const Point<1> key(y * numpcx + x);
          const coord_t x2 = x - space_rect.lo[0];
          const coord_t y2 = y - space_rect.lo[1];
          const coord_t index = y2 * ((space_rect.hi[0] - space_rect.lo[0]) + 1) + x2;
          sharding_memories[key] = space_memories[index % space_memories.size()];
        }
      }
    } else {
      sharding_memories = sharding_sys_memories;
    }
  }
  sharded = true;
  runtime->enable_reentrant(ctx);
}
#else
coord_t PennantMapper::compute_shard_index(Point<1> p)
{
  if (!sharded) {
    assert(numpcx > 0);
    assert(numpcy > 0);
    // these are member variables if we are disabling control replication
    coord_t nsx, nsy;
    calc_pieces_helper(total_nodes/*number of shards*/, numpcx, numpcy, nsx, nsy);
    // Figure out which shard we are in x and y
    const Point<2> shard_point(node_id % nsx, node_id / nsx);
    // these are member variables if we are disabling control replication
    const coord_t pershardx = (numpcx + nsx - 1) / nsx;
    const coord_t pershardy = (numpcy + nsy - 1) / nsy;
    const Rect<2> rect(Point<2>(shard_point[0] * pershardx, shard_point[1] * pershardy),
                       Point<2>((shard_point[0] + 1) * pershardx - 1,
                                (shard_point[1] + 1) * pershardy - 1));
    const Rect<2> full(Point<2>(0, 0), Point<2>(numpcx-1, numpcy-1));
    shard_rect = rect.intersection(full);
    sharded = true;
  }
  const Point<2> point(p[0] % numpcx, p[0] / numpcx);
  assert(shard_rect.contains(point));
  const coord_t x = point[0] - shard_rect.lo[0];
  const coord_t y = point[1] - shard_rect.lo[1];
  return y * ((shard_rect.hi[0] - shard_rect.lo[0]) + 1) + x;
}
#endif

