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
    pennant_mapper_name(get_name(p))
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

void PennantMapper::select_tunable_value(const MapperContext ctx,
                                         const Task& task,
                                         const SelectTunableInput& input,
                                               SelectTunableOutput& output)
{
  // No custom penant tunable values yet
  DefaultMapper::select_tunable_value(ctx, task, input, output);
}

void PennantMapper::slice_task(const Legion::Mapping::MapperContext ctx,
                               const Legion::Task &task,
                               const SliceTaskInput &input,
                                     SliceTaskOutput &output)
{
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
    output.target_procs = local_omps;
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
        create_reduction_instances(ctx, task, idx, local_zerocopy,
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
        create_reduction_instances(ctx, task, idx, local_sysmem,
                                   output.chosen_instances[idx]);
      else
        map_pennant_array(ctx, task, idx, task.regions[idx].region, 
                          local_sysmem, output.chosen_instances[idx]);
    }
  }
  runtime->acquire_instances(ctx, output.chosen_instances);
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

#ifdef CTRL_REPL
void PennantMapper::select_sharding_functor(const MapperContext ctx,
                                            const Task &task,
                                            const SelectShardingFunctorInput &input,
                                                  SelectShardingFunctorOutput &output)
{
  output.chosen_functor = PENNANT_SHARD_ID; 
}
#endif

void PennantMapper::map_copy(const MapperContext ctx,
                             const Copy &copy,
                             const MapCopyInput &input,
                                   MapCopyOutput &output)
{
  output.src_instances.resize(copy.src_requirements.size());
  output.dst_instances.resize(copy.dst_requirements.size());
  if (!local_gpus.empty()) {
    assert(copy.is_index_space);
    const coord_t point = copy.index_point[0];
    const unsigned index = point % local_gpus.size();
    const Processor gpu = local_gpus[index];
    const Memory fbmem = 
      default_policy_select_target_memory(ctx, gpu, copy.src_requirements[0]);
    for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++)
      map_pennant_array(ctx, copy, idx, copy.src_requirements[idx].region, fbmem,
                        output.src_instances[idx]);
    for (unsigned idx = 0; idx < copy.dst_requirements.size(); idx++)
      map_pennant_array(ctx, copy, idx + copy.src_requirements.size(),
                        copy.dst_requirements[idx].region, fbmem,
                        output.dst_instances[idx]);
  } else {
    for (unsigned idx = 0; idx < copy.src_requirements.size(); idx++)
      map_pennant_array(ctx, copy, idx, copy.src_requirements[idx].region, local_sysmem,
                        output.src_instances[idx]);
    for (unsigned idx = 0; idx < copy.dst_requirements.size(); idx++)
      map_pennant_array(ctx, copy, idx + copy.src_requirements.size(), 
                        copy.dst_requirements[idx].region, local_sysmem,
                        output.dst_instances[idx]);
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

#ifdef CTRL_REPL
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
  map_pennant_array(ctx, partition, 0, partition.requirement.region, local_sysmem,
                    output.chosen_instances);
  runtime->acquire_instances(ctx, output.chosen_instances);
}

/*static*/ const char* PennantMapper::get_name(Processor p)
{
  char *result = (char*)malloc(256);
  snprintf(result, 256, "Pennant Mapper on Processor %llx", p.id);
  return result;
}

#ifdef CTRL_REPL
void PennantMapper::select_sharding_functor(const MapperContext ctx,
                                            const Partition &partition,
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

