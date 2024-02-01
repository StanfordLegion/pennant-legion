/*
 * PennantMapper.hh
 *
 *  Created on: Jan 9, 2015
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef PENNANTMAPPER_HH_
#define PENNANTMAPPER_HH_

#include <set>
#include <vector>

#include "legion.h"
#include "default_mapper.h"


class PennantMapper : public Legion::Mapping::DefaultMapper {
public:
  enum {
    PREFER_CPU        = 0x0001,
    PREFER_OMP        = 0x0002,
    PREFER_GPU        = 0x0004,
    PREFER_ZCOPY      = 0x0008,
    CRITICAL          = 0x0010,
  };
public:
  PennantMapper(
        Legion::Machine machine,
        Legion::Runtime *rt,
        Legion::Processor local);
  virtual ~PennantMapper(void);
public:
  virtual const char* get_mapper_name(void) const;
  virtual Legion::Mapping::Mapper::MapperSyncModel get_mapper_sync_model(void) const;
  virtual bool request_valid_instances(void) const;
public:
  virtual void select_tunable_value(const Legion::Mapping::MapperContext ctx,
                                    const Legion::Task& task,
                                    const SelectTunableInput& input,
                                          SelectTunableOutput& output);
public:
  virtual void select_task_options(const Legion::Mapping::MapperContext ctx,
                                   const Legion::Task &task,
                                         TaskOptions &output);
  virtual Legion::Processor default_policy_select_initial_processor(
                                Legion::Mapping::MapperContext ctx, 
                                const Legion::Task &task);
  virtual void slice_task(const Legion::Mapping::MapperContext ctx,
                          const Legion::Task &task,
                          const SliceTaskInput &input,
                                SliceTaskOutput &output);
  virtual void map_task(const Legion::Mapping::MapperContext ctx,
                        const Legion::Task &task,
                        const MapTaskInput &input,
                              MapTaskOutput &output);
  // Default mapper does the right thing for map_replicate_task
  virtual void speculate(const Legion::Mapping::MapperContext ctx,
                         const Legion::Task &task,
                               SpeculativeOutput &output);
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       const Legion::Task &task,
                                       const SelectShardingFunctorInput &input,
                                             SelectShardingFunctorOutput &output);
public:
  virtual void map_copy(const Legion::Mapping::MapperContext ctx,
                        const Legion::Copy &copy,
                        const MapCopyInput &input,
                              MapCopyOutput &output);
  virtual void speculate(const Legion::Mapping::MapperContext ctx,
                         const Legion::Copy &copy,
                               SpeculativeOutput &output);
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       const Legion::Copy &copy,
                                       const SelectShardingFunctorInput &input,
                                             SelectShardingFunctorOutput &output);
public:
  virtual void select_partition_projection(const Legion::Mapping::MapperContext  ctx,
                                           const Legion::Partition& partition,
                                           const SelectPartitionProjectionInput& input,
                                                 SelectPartitionProjectionOutput& output);
  virtual void map_partition(const Legion::Mapping::MapperContext ctx,
                             const Legion::Partition& partition,
                             const MapPartitionInput&   input,
                                   MapPartitionOutput&  output);
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       const Legion::Partition &partition,
                                       const SelectShardingFunctorInput &input,
                                             SelectShardingFunctorOutput &output);
public:
  virtual void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                       const Legion::Fill& fill,
                                       const SelectShardingFunctorInput &input,
                                             SelectShardingFunctorOutput &output);
public:
  virtual void memoize_operation(const Legion::Mapping::MapperContext ctx,
                                 const Legion::Mappable& mappable,
                                 const MemoizeInput& input,
                                       MemoizeOutput& output);
protected:
  static const char* get_name(Legion::Processor p);
  void map_pennant_array(const Legion::Mapping::MapperContext ctx, 
                         const Legion::Mappable &mapple, unsigned index,
                         Legion::LogicalRegion region, Legion::Memory target,
                         std::vector<Legion::Mapping::PhysicalInstance> &instances,
                         bool initialization_instance = false);
  void create_reduction_instances(const Legion::Mapping::MapperContext ctx,
                         const Legion::Task &task, unsigned index, Legion::Memory target,
                         std::vector<Legion::Mapping::PhysicalInstance> &instances);
  Legion::VariantID find_cpu_variant(const Legion::Mapping::MapperContext ctx,
                                     Legion::TaskID task_id);
  Legion::VariantID find_omp_variant(const Legion::Mapping::MapperContext ctx,
                                     Legion::TaskID task_id);
  Legion::VariantID find_gpu_variant(const Legion::Mapping::MapperContext ctx,
                                     Legion::TaskID task_id);
#ifdef PENNANT_DISABLE_CONTROL_REPLICATION
  void compute_fake_sharding(Legion::Mapping::MapperContext ctx);
#else
  Legion::coord_t compute_shard_index(Legion::Point<1> point);
#endif
public:
  void update_mesh_information(Legion::coord_t numpcx, Legion::coord_t numpcy);
public:
  const char *const pennant_mapper_name;
protected:
  std::map<Legion::TaskID,Legion::VariantID> cpu_variants;
  std::map<Legion::TaskID,Legion::VariantID> omp_variants;
  std::map<Legion::TaskID,Legion::VariantID> gpu_variants;
protected:
  Legion::Memory local_sysmem, local_numa, local_zerocopy, local_framebuffer;
  std::map<std::pair<Legion::LogicalRegion,Legion::Memory>,
           Legion::Mapping::PhysicalInstance> local_instances;
#ifdef PENNANT_DISABLE_CONTROL_REPLICATION
  std::vector<std::pair<Legion::Processor,Legion::IndexSpace> > sharding_spaces;
  std::map<Legion::Point<1>,Legion::Memory> sharding_memories, sharding_sys_memories;
#endif
protected:
  // For helping with sharded copy operations
  Legion::coord_t numpcx, numpcy;
#ifdef PENNANT_DISABLE_CONTROL_REPLICATION
  Legion::coord_t nsx, nsy, pershardx, pershardy;
#else
  Legion::Rect<2> shard_rect;
#endif
  bool sharded;
};


#endif /* PENNANTMAPPER_HH_ */
