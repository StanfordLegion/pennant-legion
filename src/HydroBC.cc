/*
 * HydroBC.cc
 *
 *  Created on: Jan 13, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "HydroBC.hh"

#include "legion.h"

#include "MyLegion.hh"
#include "Memory.hh"
#include "Mesh.hh"
#include "Hydro.hh"

using namespace std;
using namespace Memory;
using namespace Legion;


namespace {  // unnamed
static void __attribute__ ((constructor)) registerTasks() {
  {
    TaskVariantRegistrar registrar(TID_APPLYFIXEDBC, "CPU applyfixedbc");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<HydroBC::applyFixedBCTask>(registrar, "applyfixedbc");
  }
  {
    TaskVariantRegistrar registrar(TID_APPLYFIXEDBC, "OMP applyfixedbc");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<HydroBC::applyFixedBCOMPTask>(registrar, "applyfixedbc");
  }
  {
    TaskVariantRegistrar registrar(TID_COUNTBCPOINTS, "CPU count BC points");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<HydroBC::countBCPointsTask>(registrar, "count BC points");
  }
  {
    TaskVariantRegistrar registrar(TID_COUNTBCRANGES, "CPU count BC ranges");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<coord_t, HydroBC::countBCRangesTask>(registrar, "count BC ranges");
  }
  {
    TaskVariantRegistrar registrar(TID_CREATEBCMAPS, "CPU create BC maps");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<HydroBC::createBCMapsTask>(registrar, "create BC maps");
  }
}
}; // namespace


HydroBC::HydroBC(
        Mesh* msh,
        const double2 v,
        const double bound,
        const bool xplane)
  : mesh(msh), vfix(v) {
  Context ctx = mesh->ctx;
  Runtime* runtime = mesh->runtime;
  IndexSpace is_piece = mesh->ispc;
  IndexPartition ip_piece = mesh->ippc;
  LogicalRegion lrp = mesh->lrp;
  LogicalPartition lppprv = mesh->lppprv;
  LogicalPartition lppmstr = mesh->lppmstr;
  // First compute how many points we have in this boundary condition 
  FieldSpace fsc = runtime->create_field_space(ctx); 
  {
    FieldAllocator fac = runtime->create_field_allocator(ctx, fsc); 
    fac.allocate_field(sizeof(coord_t), FID_COUNT);
    fac.allocate_field(sizeof(Rect<1>), FID_RANGE);
  }
  LogicalRegion lrc = runtime->create_logical_region(ctx, is_piece, fsc);
  LogicalPartition lpc = runtime->get_logical_partition(lrc, ip_piece);
  const double eps = 1.e-12;
  // Count how many boundary points are in each piece
  {
    const CountBCArgs args(bound, eps, xplane);
    IndexTaskLauncher launcher(TID_COUNTBCPOINTS, is_piece,
          TaskArgument(&args, sizeof(args)), ArgumentMap());
    launcher.add_region_requirement(
        RegionRequirement(lppprv, 0/*identity*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lrp));
    launcher.add_field(0/*index*/, FID_PX);
    launcher.add_region_requirement(
        RegionRequirement(lppmstr, 0/*identity*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lrp));
    launcher.add_field(1/*index*/, FID_PX);
    launcher.add_region_requirement(
        RegionRequirement(lpc, 0/*identity*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lrc));
    launcher.add_field(2/*index*/, FID_COUNT);
    runtime->execute_index_space(ctx, launcher);
  }
  // Construct the ranges
  {
    TaskLauncher launcher(TID_COUNTBCRANGES, TaskArgument());
    launcher.add_region_requirement(
        RegionRequirement(lrc, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lrc));
    launcher.add_field(0/*index*/, FID_COUNT);
    launcher.add_region_requirement(
        RegionRequirement(lrc, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lrc));
    launcher.add_field(1/*index*/, FID_RANGE);
    Future f = runtime->execute_task(ctx, launcher);
    numb = f.get_result<coord_t>(true/*silence warnings*/);
  }
  // Make the index space and compute the partition of it for the pieces
  IndexSpace isb = runtime->create_index_space(ctx, Rect<1>(0,numb-1));
  FieldSpace fsb = runtime->create_field_space(ctx);
  {
    FieldAllocator fab = runtime->create_field_allocator(ctx, fsb);
    fab.allocate_field(sizeof(Pointer), FID_MAPBP);
    fab.allocate_field(sizeof(int), FID_MAPBPREG);
  }
  lrb = runtime->create_logical_region(ctx, isb, fsb);
  IndexPartition ipb = 
    runtime->create_partition_by_image_range(ctx, isb, 
                        lpc, lrc, FID_RANGE, is_piece);
  lpb = runtime->get_logical_partition(lrb, ipb);
  // Then fill in the mapping to points and their location
  {
    const CountBCArgs args(bound, eps, xplane);
    IndexTaskLauncher launcher(TID_CREATEBCMAPS, is_piece,
        TaskArgument(&args, sizeof(args)), ArgumentMap());
    launcher.add_region_requirement(
        RegionRequirement(lppprv, 0/*identity*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lrp));
    launcher.add_field(0/*index*/, FID_PX);
    launcher.add_region_requirement(
        RegionRequirement(lppmstr, 0/*identity*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lrp));
    launcher.add_field(1/*index*/, FID_PX);
    launcher.add_region_requirement(
        RegionRequirement(lpb, 0/*identity*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lrb));
    launcher.add_field(2/*index*/, FID_MAPBP);
    launcher.add_field(2/*index*/, FID_MAPBPREG);
    runtime->execute_index_space(ctx, launcher);
  }
}


HydroBC::~HydroBC() {}


void HydroBC::applyFixedBCTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double2* args = (const double2*) task->args;
    const double2 vfix = args[0];
    const AccessorRO<Pointer> acc_mapbp(regions[0], FID_MAPBP);
    const AccessorRO<int> acc_mapbpreg(regions[0], FID_MAPBPREG);
    const AccessorRW<double2> acc_pf[2] = {
        AccessorRW<double2>(regions[1], FID_PF),
        AccessorRW<double2>(regions[2], FID_PF)
    };
    const AccessorRW<double2> acc_pu[2] = {
        AccessorRW<double2>(regions[1], FID_PU0),
        AccessorRW<double2>(regions[2], FID_PU0)
    };

    const IndexSpace& isb = task->regions[0].region.get_index_space();
   
    for (PointIterator itb(runtime, isb); itb(); itb++)
    {
        const Pointer p = acc_mapbp[*itb];
        const int preg = acc_mapbpreg[*itb];
        double2 pu = acc_pu[preg][p];
        double2 pf = acc_pf[preg][p];
        pu = project(pu, vfix);
        pf = project(pf, vfix);
        acc_pu[preg][p] = pu;
        acc_pf[preg][p] = pf;
    }

}


void HydroBC::applyFixedBCOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double2* args = (const double2*) task->args;
    const double2 vfix = args[0];
    const AccessorRO<Pointer> acc_mapbp(regions[0], FID_MAPBP);
    const AccessorRO<int> acc_mapbpreg(regions[0], FID_MAPBPREG);
    const AccessorRW<double2> acc_pf[2] = {
        AccessorRW<double2>(regions[1], FID_PF),
        AccessorRW<double2>(regions[2], FID_PF)
    };
    const AccessorRW<double2> acc_pu[2] = {
        AccessorRW<double2>(regions[1], FID_PU0),
        AccessorRW<double2>(regions[2], FID_PU0)
    };

    const IndexSpace& isb = task->regions[0].region.get_index_space();
    // This will fail if it is not dense
    const Rect<1> rectb = runtime->get_index_space_domain(isb);
    #pragma omp parallel for
    for (coord_t b = rectb.lo[0]; b <= rectb.hi[0]; b++)
    {
        const Pointer p = acc_mapbp[b];
        const int preg = acc_mapbpreg[b];
        double2 pu = acc_pu[preg][p];
        double2 pf = acc_pf[preg][p];
        pu = project(pu, vfix);
        pf = project(pf, vfix);
        acc_pu[preg][p] = pu;
        acc_pf[preg][p] = pf;
    }

}


void HydroBC::countBCPointsTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const CountBCArgs *args = reinterpret_cast<const CountBCArgs*>(task->args);
    const AccessorRO<double2> acc_priv(regions[0], FID_PX);
    const AccessorRO<double2> acc_mstr(regions[1], FID_PX);
    const AccessorWD<coord_t> acc_cnt(regions[2], FID_COUNT);

    IndexSpace is_priv = task->regions[0].region.get_index_space();
    IndexSpace is_mstr = task->regions[1].region.get_index_space();

    coord_t count = 0;
    if (args->xplane) {
      for (PointIterator itr(runtime, is_priv); itr(); itr++)
        if (fabs(acc_priv[*itr].x - args->bound) < args->eps)
          count++;
      for (PointIterator itr(runtime, is_mstr); itr(); itr++)
        if (fabs(acc_mstr[*itr].x - args->bound) < args->eps)
          count++;
    } else {
      for (PointIterator itr(runtime, is_priv); itr(); itr++)
        if (fabs(acc_priv[*itr].y - args->bound) < args->eps)
          count++;
      for (PointIterator itr(runtime, is_mstr); itr(); itr++)
        if (fabs(acc_mstr[*itr].y - args->bound) < args->eps)
          count++;
    }
    acc_cnt[task->index_point] = count;
}


coord_t HydroBC::countBCRangesTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<coord_t>  acc_count(regions[0], FID_COUNT);
    const AccessorWD<Rect<1> > acc_range(regions[1], FID_RANGE);
    
    IndexSpace is_piece = task->regions[0].region.get_index_space();
    coord_t current = 0;
    for (PointIterator itr(runtime, is_piece); itr(); itr++)
    {
      const coord_t count = acc_count[*itr];
      acc_range[*itr] = Rect<1>(current, current + count - 1);
      current += count;
    }
    return current;
}


void HydroBC::createBCMapsTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const CountBCArgs *args = reinterpret_cast<const CountBCArgs*>(task->args);
    const AccessorRO<double2> acc_priv(regions[0], FID_PX);
    const AccessorRO<double2> acc_mstr(regions[1], FID_PX);
    const AccessorWD<Pointer> acc_ptr(regions[2], FID_MAPBP);
    const AccessorWD<int>     acc_reg(regions[2], FID_MAPBPREG);

    IndexSpace is_priv = task->regions[0].region.get_index_space();
    IndexSpace is_mstr = task->regions[1].region.get_index_space();
    IndexSpace is_out  = task->regions[2].region.get_index_space();

    PointIterator out_itr(runtime, is_out);
    if (args->xplane) {
      for (PointIterator itr(runtime, is_priv); itr(); itr++)
        if (fabs(acc_priv[*itr].x - args->bound) < args->eps)
        {
          assert(out_itr());
          acc_ptr[*out_itr] = *itr;
          acc_reg[*out_itr] = 0;
          out_itr++;
        }
      for (PointIterator itr(runtime, is_mstr); itr(); itr++)
        if (fabs(acc_mstr[*itr].x - args->bound) < args->eps)
        {
          assert(out_itr());
          acc_ptr[*out_itr] = *itr;
          acc_reg[*out_itr] = 1;
          out_itr++;
        }
    } else {
      for (PointIterator itr(runtime, is_priv); itr(); itr++)
        if (fabs(acc_priv[*itr].y - args->bound) < args->eps)
        {
          assert(out_itr());
          acc_ptr[*out_itr] = *itr;
          acc_reg[*out_itr] = 0;
          out_itr++;
        }
      for (PointIterator itr(runtime, is_mstr); itr(); itr++)
        if (fabs(acc_mstr[*itr].y - args->bound) < args->eps)
        {
          assert(out_itr());
          acc_ptr[*out_itr] = *itr;
          acc_reg[*out_itr] = 1;
          out_itr++;
        }
    }
}

