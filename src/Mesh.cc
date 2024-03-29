/*
 * Mesh.cc
 *
 *  Created on: Jan 5, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "Mesh.hh"

#include <cmath>
#include <iostream>
#include <algorithm>
#include <float.h>

#include "legion.h"

#include "MyLegion.hh"
#include "Vec2.hh"
#include "Memory.hh"
#include "InputFile.hh"
#include "GenMesh.hh"
#include "WriteXY.hh"
#include "ExportGold.hh"
#include "PennantMapper.hh"

using namespace std;
using namespace Memory;
using namespace Legion;


namespace {  // unnamed
static void __attribute__ ((constructor)) registerTasks() {
    {
      TaskVariantRegistrar registrar(TID_CALCCTRS, "CPU calcctrs");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcCtrsTask>(registrar, "calcctrs");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCCTRS, "OMP calcctrs");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcCtrsOMPTask>(registrar, "calcctrs");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCVOLS, "CPU calcvols");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<int, Mesh::calcVolsTask>(registrar, "calcvols");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCVOLS, "OMP calcvols");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<int, Mesh::calcVolsOMPTask>(registrar, "calcvols");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCSIDEFRACS, "CPU calcsidefracs");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcSideFracsTask>(registrar, "sidefracs");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCSURFVECS, "CPU calcsurfvecs");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcSurfVecsTask>(registrar, "calcsurfvecs");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCSURFVECS, "OMP calcsurfvecs");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcSurfVecsOMPTask>(registrar, "calcsurfvecs");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCEDGELEN, "CPU calcedgelen");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcEdgeLenTask>(registrar, "calcedgelen");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCEDGELEN, "OMP calcedgelen");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcEdgeLenOMPTask>(registrar, "calcedgelen");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCCHARLEN, "CPU calccharlen");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcCharLenTask>(registrar, "calccharlen");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCCHARLEN, "CPU calccharlen");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcCharLenOMPTask>(registrar, "calccharlen");
    }
    {
      TaskVariantRegistrar registrar(TID_COUNTPOINTS, "CPU count points");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::countPointsTask>(registrar, "count points");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCRANGES, "CPU calc ranges");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcRangesTask>(registrar, "calc ranges");
    }
    {
      TaskVariantRegistrar registrar(TID_COMPACTPOINTS, "CPU compact points");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::compactPointsTask>(registrar, "compact points");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCOWNERS, "CPU calc owners");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcOwnersTask>(registrar, "calc owners");
    }
    {
      TaskVariantRegistrar registrar(TID_CHECKBADSIDES, "CPU check bad sides");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::checkBadSidesTask>(registrar, "check bad sides");
    }
    {
      TaskVariantRegistrar registrar(TID_TEMPGATHER, "CPU temp gather");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::tempGatherTask>(registrar, "temp gather");
    }
    {
      TaskVariantRegistrar registrar(TID_WRITE, "CPU write out");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::writeTask>(registrar, "write out");
    }

    Runtime::register_reduction_op<SumOp<int> >(
            OPID_SUMINT);
    Runtime::register_reduction_op<SumOp<double> >(
            OPID_SUMDBL);
    Runtime::register_reduction_op<SumOp<double2> >(
            OPID_SUMDBL2);
    Runtime::register_reduction_op<MinOp<double> >(
            OPID_MINDBL);
    Runtime::register_reduction_op<MaxOp<double> >(
            OPID_MAXDBL);
}
}; // namespace


template <>
const int SumOp<int>::identity = 0;
template <>
const double SumOp<double>::identity = 0.;
template <>
const double2 SumOp<double2>::identity = double2(0., 0.);
template <>
const double MinOp<double>::identity = DBL_MAX;
template <>
const double MaxOp<double>::identity = DBL_MIN;

PennantShardingFunctor::PennantShardingFunctor(const coord_t nx, const coord_t ny)
  : ShardingFunctor(), numpcx(nx), numpcy(ny), sharded(false) { }

ShardID PennantShardingFunctor::shard(const DomainPoint &p,
                                      const Domain &full_space,
                                      const size_t total_shards)
{
  if (!sharded) {
    // Figure out our sharding information
    // Tile pieces on shards the same way we tile
    // pieces on the original mesh
    // Check that we're doing this for the right thing
    const Rect<1> space = full_space;
    assert((numpcx * numpcy) == space.volume());
    calc_pieces_helper(total_shards, numpcx, numpcy, nsx, nsy);
    pershardx = (numpcx + nsx - 1) / nsx;
    pershardy = (numpcy + nsy - 1) / nsy;
    // When we're done we can save everything
    shards = (coord_t)total_shards;
    // Make sure all the writes are flushed
    __sync_synchronize();
    sharded = true;
  } else {
    // Sanity check we're using the right thing
    assert(shards == ((coord_t)total_shards));
  }
  // Then we can compute our point information
  const Point<1> point = p; 
  const coord_t pcx = point % numpcx;
  const coord_t pcy = point / numpcx;
  const coord_t sx = pcx / pershardx;
  const coord_t sy = pcy / pershardy;
  ShardID result = sy * nsx + sx;
  assert(result < total_shards);
  return result;
}

/*static*/ std::vector<PennantMapper*> Mesh::local_mappers;

Mesh::Mesh(
        const InputFile* inp,
        const int numpcsa,
        Context ctxa,
        Runtime* runtimea)
        : gmesh(NULL), numpcs(numpcsa), ctx(ctxa), runtime(runtimea) {

    chunksize = inp->getInt("chunksize", 0);
    subregion = inp->getDoubleList("subregion", vector<double>());
    if (subregion.size() != 0 && subregion.size() != 4) {
        cerr << "Error:  subregion must have 4 entries" << endl;
        exit(1);
    }

    gmesh = new GenMesh(inp);

    // Call this to populate the numpcx and numpcy fields
    gmesh->calcNumPieces(numpcs);
    PennantShardingFunctor *functor = 
      new PennantShardingFunctor(gmesh->numpcx, gmesh->numpcy);
    runtime->register_sharding_functor(PENNANT_SHARD_ID, functor, 
                                       true/*silence warnings*/);
    // This is a little bit brittle but for now we need to tell our
    // mappers about the size of the mesh which we are about to 
    // load so that it knows how to shard things
    for (std::vector<PennantMapper*>::const_iterator it = 
          local_mappers.begin(); it != local_mappers.end(); it++)
      (*it)->update_mesh_information(gmesh->numpcx, gmesh->numpcy);

    init();
}


Mesh::~Mesh() {
    delete gmesh;
}


void Mesh::init() {
    const Rect<1> piece_rect(Point<1>(0), Point<1>(numpcs-1));
    dompc  = Domain(piece_rect);
    // Create a space for the number of pieces that we will have
    IndexSpace is_piece = runtime->create_index_space(ctx, piece_rect);
    this->ispc = is_piece;
    IndexPartition ip_piece = runtime->create_equal_partition(ctx, is_piece, is_piece);
    ippc = ip_piece;

    // Create point index space and field spaces
    nump = gmesh->calcNumPoints(numpcs);
    IndexSpace isp = runtime->create_index_space(ctx, Rect<1>(0,nump-1));
    FieldSpace fsp = runtime->create_field_space(ctx);
    {
      FieldAllocator fap = runtime->create_field_allocator(ctx, fsp);
      fap.allocate_field(sizeof(double2), FID_PX);
      runtime->attach_name(fsp, FID_PX, "PX");
      fap.allocate_field(sizeof(double2), FID_PXP);
      runtime->attach_name(fsp, FID_PXP, "PXP");
      fap.allocate_field(sizeof(double2), FID_PX0);
      runtime->attach_name(fsp, FID_PX0, "PX0");
      fap.allocate_field(sizeof(double2), FID_PU);
      runtime->attach_name(fsp, FID_PU, "PU");
      fap.allocate_field(sizeof(double2), FID_PU0);
      runtime->attach_name(fsp, FID_PU0, "PU0");
      fap.allocate_field(sizeof(double), FID_PMASWT);
      runtime->attach_name(fsp, FID_PMASWT, "PMASWT");
      fap.allocate_field(sizeof(double2), FID_PF);
      runtime->attach_name(fsp, FID_PF, "PF");
      fap.allocate_field(sizeof(double2), FID_PAP);
      runtime->attach_name(fsp, FID_PAP, "PAP");
      fap.allocate_field(sizeof(coord_t), FID_PIECE);
      runtime->attach_name(fsp, FID_PIECE, "PIECE");
      fap.allocate_field(sizeof(Pointer), FID_MAPLOAD2DENSE);
      runtime->attach_name(fsp, FID_MAPLOAD2DENSE, "MAPLOAD2DENSE");
    }

    // load fields into temp points with equal partition
    LogicalRegion lr_temp_points = runtime->create_logical_region(ctx, isp, fsp);
    IndexPartition ip_points_equal = runtime->create_equal_partition(ctx, isp, is_piece);
    LogicalPartition lp_points_equal = 
      runtime->get_logical_partition(lr_temp_points, ip_points_equal);
    gmesh->generatePointsParallel(numpcs, runtime, ctx, 
                                  lr_temp_points, lp_points_equal, is_piece); 

    // equal partition zones
    numz = gmesh->calcNumZones(numpcs);
    IndexSpace isz = runtime->create_index_space(ctx, Rect<1>(0, numz-1));
    FieldSpace fsz = runtime->create_field_space(ctx);
    {
      FieldAllocator faz = runtime->create_field_allocator(ctx, fsz);
      faz.allocate_field(sizeof(int), FID_ZNUMP);
      runtime->attach_name(fsz, FID_ZNUMP, "ZNUMP");
      faz.allocate_field(sizeof(double2), FID_ZX);
      runtime->attach_name(fsz, FID_ZX, "ZX");
      faz.allocate_field(sizeof(double2), FID_ZXP);
      runtime->attach_name(fsz, FID_ZXP, "ZXP");
      faz.allocate_field(sizeof(double), FID_ZAREA);
      runtime->attach_name(fsz, FID_ZAREA, "ZAREA");
      faz.allocate_field(sizeof(double), FID_ZVOL);
      runtime->attach_name(fsz, FID_ZVOL, "ZVOL");
      faz.allocate_field(sizeof(double), FID_ZAREAP);
      runtime->attach_name(fsz, FID_ZAREAP, "ZAREAP");
      faz.allocate_field(sizeof(double), FID_ZVOLP);
      runtime->attach_name(fsz, FID_ZVOLP, "ZVOLP");
      faz.allocate_field(sizeof(double), FID_ZVOL0);
      runtime->attach_name(fsz, FID_ZVOL0, "ZVOL0");
      faz.allocate_field(sizeof(double), FID_ZDL);
      runtime->attach_name(fsz, FID_ZDL, "ZDL");
      faz.allocate_field(sizeof(double), FID_ZM);
      runtime->attach_name(fsz, FID_ZM, "ZM");
      faz.allocate_field(sizeof(double), FID_ZR);
      runtime->attach_name(fsz, FID_ZR, "ZR");
      faz.allocate_field(sizeof(double), FID_ZRP);
      runtime->attach_name(fsz, FID_ZRP, "ZRP");
      faz.allocate_field(sizeof(double), FID_ZE);
      runtime->attach_name(fsz, FID_ZE, "ZE");
      faz.allocate_field(sizeof(double), FID_ZETOT);
      runtime->attach_name(fsz, FID_ZETOT, "ZETOT");
      faz.allocate_field(sizeof(double), FID_ZW);
      runtime->attach_name(fsz, FID_ZW, "ZW");
      faz.allocate_field(sizeof(double), FID_ZWRATE);
      runtime->attach_name(fsz, FID_ZWRATE, "ZWRATE");
      faz.allocate_field(sizeof(double), FID_ZP);
      runtime->attach_name(fsz, FID_ZP, "ZP");
      faz.allocate_field(sizeof(double), FID_ZSS);
      runtime->attach_name(fsz, FID_ZSS, "ZSS");
      faz.allocate_field(sizeof(double), FID_ZDU);
      runtime->attach_name(fsz, FID_ZDU, "ZDU");
      faz.allocate_field(sizeof(double2), FID_ZUC);
      runtime->attach_name(fsz, FID_ZUC, "ZUC");
      faz.allocate_field(sizeof(double), FID_ZTMP);
      runtime->attach_name(fsz, FID_ZTMP, "ZTMP");
      faz.allocate_field(sizeof(Pointer), FID_PIECE);
      runtime->attach_name(fsz, FID_PIECE, "PIECE");
    }
    lrz = runtime->create_logical_region(ctx, isz, fsz);
    runtime->attach_name(lrz, "lrz");
    IndexPartition zones_equal = runtime->create_equal_partition(ctx, isz, is_piece);
    // fill in the number of sides for each zone
    gmesh->generateZonesParallel(numpcs, runtime, ctx, lrz, 
        runtime->get_logical_partition(ctx, lrz, zones_equal), is_piece);

    // Create sides logical region
    nums = gmesh->calcNumSides(numpcs);
    numc = nums;
    IndexSpace iss = runtime->create_index_space(ctx, Rect<1>(0, nums-1));
    FieldSpace fss = runtime->create_field_space(ctx);
    {
      FieldAllocator fas = runtime->create_field_allocator(ctx, fss);
      fas.allocate_field(sizeof(Pointer), FID_MAPSP1);
      runtime->attach_name(fss, FID_MAPSP1, "MAPSP1");
#ifndef PRECOMPACTED_RECT_POINTS
      fas.allocate_field(sizeof(Pointer), FID_MAPSP1TEMP);
      runtime->attach_name(fss, FID_MAPSP1TEMP, "MAPSP1TEMP");
#endif
      fas.allocate_field(sizeof(Pointer), FID_MAPSP2);
      runtime->attach_name(fss, FID_MAPSP2, "MAPSP2");
#ifndef PRECOMPACTED_RECT_POINTS
      fas.allocate_field(sizeof(Pointer), FID_MAPSP2TEMP);
      runtime->attach_name(fss, FID_MAPSP2TEMP, "MAPSP2TEMP");
#endif
      fas.allocate_field(sizeof(Pointer), FID_MAPSZ);
      runtime->attach_name(fss, FID_MAPSZ, "MAPSZ");
      fas.allocate_field(sizeof(Pointer), FID_MAPSS3);
      runtime->attach_name(fss, FID_MAPSS3, "MAPSS3");
      fas.allocate_field(sizeof(Pointer), FID_MAPSS4);
      runtime->attach_name(fss, FID_MAPSS4, "MAPSS4");
      fas.allocate_field(sizeof(int), FID_MAPSP1REG);
      runtime->attach_name(fss, FID_MAPSP1REG, "MAPSP1REG");
      fas.allocate_field(sizeof(int), FID_MAPSP2REG);
      runtime->attach_name(fss, FID_MAPSP2REG, "MAPSP2REG");
      fas.allocate_field(sizeof(double2), FID_EX);
      runtime->attach_name(fss, FID_EX, "EX");
      fas.allocate_field(sizeof(double2), FID_EXP);
      runtime->attach_name(fss, FID_EXP, "EXP");
      fas.allocate_field(sizeof(double), FID_SAREA);
      runtime->attach_name(fss, FID_SAREA, "SAREA");
      fas.allocate_field(sizeof(double), FID_SVOL);
      runtime->attach_name(fss, FID_SVOL, "SVOL");
      fas.allocate_field(sizeof(double), FID_SAREAP);
      runtime->attach_name(fss, FID_SAREAP, "SAREAP");
      fas.allocate_field(sizeof(double), FID_SVOLP);
      runtime->attach_name(fss, FID_SVOLP, "SVOLP");
      fas.allocate_field(sizeof(double2), FID_SSURFP);
      runtime->attach_name(fss, FID_SSURFP, "SSURFP");
      fas.allocate_field(sizeof(double), FID_ELEN);
      runtime->attach_name(fss, FID_ELEN, "ELEN");
      fas.allocate_field(sizeof(double), FID_SMF);
      runtime->attach_name(fss, FID_SMF, "SMF");
      fas.allocate_field(sizeof(double2), FID_SFP);
      runtime->attach_name(fss, FID_SFP, "SFP");
      fas.allocate_field(sizeof(double2), FID_SFQ);
      runtime->attach_name(fss, FID_SFQ, "SFQ");
      fas.allocate_field(sizeof(double2), FID_SFT);
      runtime->attach_name(fss, FID_SFT, "SFT");
      fas.allocate_field(sizeof(double), FID_CAREA);
      runtime->attach_name(fss, FID_CAREA, "CAREA");
      fas.allocate_field(sizeof(double), FID_CEVOL);
      runtime->attach_name(fss, FID_CEVOL, "CEVOL");
      fas.allocate_field(sizeof(double), FID_CDU);
      runtime->attach_name(fss, FID_CDU, "CDU");
      fas.allocate_field(sizeof(double), FID_CDIV);
      runtime->attach_name(fss, FID_CDIV, "CDIV");
      fas.allocate_field(sizeof(double), FID_CCOS);
      runtime->attach_name(fss, FID_CCOS, "CCOS");
      fas.allocate_field(sizeof(double2), FID_CQE1);
      runtime->attach_name(fss, FID_CQE1, "CQE1");
      fas.allocate_field(sizeof(double2), FID_CQE2);
      runtime->attach_name(fss, FID_CQE2, "CQE2");
      fas.allocate_field(sizeof(double), FID_CRMU);
      runtime->attach_name(fss, FID_CRMU, "CRMU");
      fas.allocate_field(sizeof(double), FID_CW);
      runtime->attach_name(fss, FID_CW, "CW");
    }
    lrs = runtime->create_logical_region(ctx, iss, fss);
    runtime->attach_name(lrs, "lrs");
    IndexPartition equal_sides = runtime->create_equal_partition(ctx, iss, is_piece);
    // construct temp side maps with equal partition (iterate over zones and find sides)
    gmesh->generateSidesParallel(numpcs, runtime, ctx, lrs, 
        runtime->get_logical_partition(lrs, equal_sides), is_piece);

    // Get the proper zone and side partitions for our pieces
    IndexPartition zone_pieces = 
      runtime->create_partition_by_field(ctx, lrz, lrz, FID_PIECE, is_piece);
    lpz = runtime->get_logical_partition(lrz, zone_pieces);
    IndexPartition side_pieces = 
      runtime->create_partition_by_preimage(ctx, zone_pieces, lrs, lrs, FID_MAPSZ, is_piece);
    lps = runtime->get_logical_partition(lrs, side_pieces);

    // Now we need to compact our points and generate our point partition tree
    // First compute our owned points
    IndexPartition ip_owned_points = runtime->create_partition_by_field(ctx, 
                                    lr_temp_points, lr_temp_points, FID_PIECE, is_piece);
    runtime->attach_name(ip_owned_points, "owned points");
    
    // Now find the set of points that we can reach from our points through all our sides
    IndexPartition ip_reachable_points = runtime->create_partition_by_image(ctx, isp,
                                                  lps, lrs, 
#ifdef PRECOMPACTED_RECT_POINTS
                                                  FID_MAPSP1,
#else
                                                  FID_MAPSP1TEMP, 
#endif
                                                  is_piece);
    runtime->attach_name(ip_reachable_points, "reachable points");

    // Now we can make the temp ghost partition
    IndexPartition ip_temp_ghost_points = runtime->create_partition_by_difference(ctx,
                                isp, ip_reachable_points, ip_owned_points, is_piece);
    runtime->attach_name(ip_temp_ghost_points, "temporary ghost points");

    // Now create a two-way partition of private versus shared
    IndexSpace is_private = runtime->create_index_space(ctx, Rect<1>(0, 1));
    IndexPartition ip_private_shared = 
      runtime->create_pending_partition(ctx, isp, is_private);

    // Fill in the two sub-regions of the ip_private_shared partition
    IndexSpace is_all_shared = runtime->create_index_space_union(ctx, 
        ip_private_shared, Point<1>(1)/*color*/, ip_temp_ghost_points);
    std::vector<IndexSpace> diff_spaces(1, is_all_shared);
    IndexSpace is_all_private = runtime->create_index_space_difference(ctx, 
                  ip_private_shared, Point<1>(0)/*color*/, isp, diff_spaces);
    runtime->attach_name(ip_private_shared, "all private-shared");

    // create the private and shared partitions with cross product partitions
    // There are only going to be two of them so we can get their names back
    // right away without having to worry about scalability
    std::map<IndexSpace,IndexPartition> partition_handles;
    partition_handles[is_all_shared] = IndexPartition::NO_PART;
    partition_handles[is_all_private] = IndexPartition::NO_PART;
    runtime->create_cross_product_partitions(ctx, ip_private_shared, 
                                ip_owned_points, partition_handles);
    IndexPartition ip_temp_master = partition_handles[is_all_shared];
    IndexPartition ip_temp_private = partition_handles[is_all_private];

    // The problem with these partitions is that the sub-regions in them
    // might not be dense, so we now need to make the actual dense partitions
    // To do this we make a temporary region of the number of pieces and count
    // how many points exist in private and shared and then use image partitions
    // to compute them
    FieldSpace fsc = runtime->create_field_space(ctx);
    {
      FieldAllocator fac = runtime->create_field_allocator(ctx, fsc); 
      fac.allocate_field(sizeof(coord_t), FID_COUNT);
      fac.allocate_field(sizeof(Rect<1>), FID_RANGE);
    }
    LogicalRegion lr_all_range = runtime->create_logical_region(ctx, is_private, fsc);
    LogicalRegion lr_private_range = runtime->create_logical_region(ctx, is_piece, fsc);
    LogicalPartition lp_private_range = 
      runtime->get_logical_partition(lr_private_range, ip_piece);
    LogicalRegion lr_shared_range = runtime->create_logical_region(ctx, is_piece, fsc);
    LogicalPartition lp_shared_range = 
      runtime->get_logical_partition(lr_shared_range, ip_piece);
    computeRangesParallel(numpcs, runtime, ctx, lr_all_range,
        lr_private_range, lp_private_range, lr_shared_range, lp_shared_range,
        ip_temp_private, ip_temp_master, is_piece);

    // Now we can compute the actual dense versions of the partition
    IndexPartition private_ip = runtime->create_equal_partition(ctx, is_private, is_private);
    IndexPartition ippall = runtime->create_partition_by_image_range(ctx, isp,
        runtime->get_logical_partition(lr_all_range, private_ip), lr_all_range, 
        FID_RANGE, is_private);
    IndexSpace is_prv = runtime->get_index_subspace(ippall, DomainPoint(0));
    IndexSpace is_shr = runtime->get_index_subspace(ippall, DomainPoint(1));
    IndexPartition ip_prv = runtime->create_partition_by_image_range(ctx, is_prv,
        lp_private_range, lr_private_range, FID_RANGE, is_piece);
    IndexPartition ip_mstr = runtime->create_partition_by_image_range(ctx, is_shr,
        lp_shared_range, lr_shared_range, FID_RANGE, is_piece);

    // Now make the actual point logical region, get the partitions, and copy over data
#ifdef PRECOMPACTED_RECT_POINTS
    // This is a very special case where we can just using the existing region
    // without needing to ever copy anything
    lrp = lr_temp_points;
    runtime->attach_name(lrp, "lrp");
    lppprv = runtime->get_logical_partition_by_tree(ip_prv, fsp, lrp.get_tree_id());
    runtime->attach_name(lppprv, "lppprv");
    lppmstr = runtime->get_logical_partition_by_tree(ip_mstr, fsp, lrp.get_tree_id());
    runtime->attach_name(lppmstr, "lppmstr");
#else
    lrp = runtime->create_logical_region(ctx, isp, fsp);
    runtime->attach_name(lrp, "lrp");
    lppprv = runtime->get_logical_partition_by_tree(ip_prv, fsp, lrp.get_tree_id());
    runtime->attach_name(lppprv, "lppprv");
    lppmstr = runtime->get_logical_partition_by_tree(ip_mstr, fsp, lrp.get_tree_id());
    runtime->attach_name(lppmstr, "lppmstr");

    // Compact the points
    compactPointsParallel(numpcs, runtime, ctx, lr_temp_points, 
        runtime->get_logical_partition_by_tree(ip_temp_private, fsp, 
          lr_temp_points.get_tree_id()), lrp, lppprv, is_piece);
    compactPointsParallel(numpcs, runtime, ctx, lr_temp_points,
        runtime->get_logical_partition_by_tree(ip_temp_master, fsp, 
          lr_temp_points.get_tree_id()), lrp, lppmstr, is_piece);

    // Update the side pointers to points with a gather copy
    // Gather copies aren't quite ready yet so we'll do this with
    // a very simple gather copy task for now, but we will switch
    // this over to proper gather copies once the runtime supports them
#ifdef ENABLE_GATHER_COPIES
    {
      IndexCopyLauncher update_launcher(is_piece);
      update_launcher.add_copy_requirements(
          RegionRequirement(lp_points_equal, 0/*identity projection*/, 
                            LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_temp_points),
          RegionRequirement(lps, 0/*identity projection*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lrs));
      update_launcher.add_src_field(0/*index*/, FID_MAPLOAD2DENSE);
      update_launcher.add_dst_field(0/*index*/, FID_MAPSP1);
      update_launcher.add_src_indirect_field(FID_MAPSP1TEMP,
          RegionRequirement(lps, 0/*identity projection*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lrs)); 
      update_launcher.possible_src_indirect_out_of_range = false;
      runtime->issue_copy_operation(ctx, update_launcher);
    }
    {
      IndexCopyLauncher update_launcher(is_piece);
      update_launcher.add_copy_requirements(
          RegionRequirement(lp_points_equal, 0/*identity projection*/, 
                            LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_temp_points),
          RegionRequirement(lps, 0/*identity projection*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lrs));
      update_launcher.add_src_field(0/*index*/, FID_MAPLOAD2DENSE);
      update_launcher.add_dst_field(0/*index*/, FID_MAPSP2);
      update_launcher.add_src_indirect_field(FID_MAPSP2TEMP,
          RegionRequirement(lps, 0/*identity projection*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lrs));
      update_launcher.possible_src_indirect_out_of_range = false;
      runtime->issue_copy_operation(ctx, update_launcher);
    }
#else
    {
      TaskLauncher update_launcher(TID_TEMPGATHER, TaskArgument());
      update_launcher.add_region_requirement(
          RegionRequirement(lr_temp_points, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_temp_points));
      update_launcher.add_field(0/*index*/, FID_MAPLOAD2DENSE);
      update_launcher.add_region_requirement(
          RegionRequirement(lrs, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lrs));
      update_launcher.add_field(1/*index*/, FID_MAPSP1);
      update_launcher.add_region_requirement(
          RegionRequirement(lrs, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lrs));
      update_launcher.add_field(2/*index*/, FID_MAPSP1TEMP);
      runtime->execute_task(ctx, update_launcher);
    }
    {
      TaskLauncher update_launcher(TID_TEMPGATHER, TaskArgument());
      update_launcher.add_region_requirement(
          RegionRequirement(lr_temp_points, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_temp_points));
      update_launcher.add_field(0/*index*/, FID_MAPLOAD2DENSE);
      update_launcher.add_region_requirement(
          RegionRequirement(lrs, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lrs));
      update_launcher.add_field(1/*index*/, FID_MAPSP2);
      update_launcher.add_region_requirement(
          RegionRequirement(lrs, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lrs));
      update_launcher.add_field(2/*index*/, FID_MAPSP2TEMP);
      runtime->execute_task(ctx, update_launcher);
    }
#endif
#endif

    // Lastly we need to get the shared partition by performing an image
    // through one of the point mappings to get the set of shared points
    // Note that this gets scoped by the is_shr index space to avoid any
    // private points
    IndexPartition ip_shr = runtime->create_partition_by_image(ctx, is_shr, 
                                            lps, lrs, FID_MAPSP1, is_piece);
    lppshr = runtime->get_logical_partition_by_tree(ip_shr, fsp, lrp.get_tree_id());
    runtime->attach_name(lppshr, "lppshr");

    // Figure out which points are private and shared for our sides
    calcOwnershipParallel(runtime, ctx, lrs, lps, ip_prv, ip_shr, is_piece);

    // Calculate centers, volumes, and side fractions
    calcCtrsParallel(runtime, ctx, lrs, lps, lrz, lpz, lrp, lppprv, lppshr, is_piece);
    Future numsbad = 
      calcVolsParallel(runtime, ctx, lrs, lps, lrz, lpz, lrp, lppprv, lppshr, is_piece);
    checkBadSides(-1/*init cycle*/, numsbad, Predicate::TRUE_PRED);
    calcSideFracsParallel(runtime, ctx, lrs, lps, lrz, lpz, is_piece);

    // create index spaces and fields for global vars
    IndexSpace isglb = runtime->create_index_space(ctx, Rect<1>(Point<1>(0),Point<1>(0)));
    FieldSpace fsglb = runtime->create_field_space(ctx);
    {
      FieldAllocator faglb = runtime->create_field_allocator(ctx, fsglb);
      faglb.allocate_field(sizeof(int), FID_NUMSBAD);
      faglb.allocate_field(sizeof(double), FID_DTREC);
    }
    lrglb = runtime->create_logical_region(ctx, isglb, fsglb);
    runtime->attach_name(lrglb, "lrglb");
    {
      FillLauncher fill(lrglb, lrglb, numsbad);
      fill.add_field(FID_NUMSBAD);
      runtime->fill_fields(ctx, fill);
    }

    // Delete our temporary regions
#ifndef PRECOMPACTED_RECT_POINTS
    runtime->destroy_logical_region(ctx, lr_temp_points);
#endif
    runtime->destroy_logical_region(ctx, lr_all_range);
    runtime->destroy_logical_region(ctx, lr_private_range);
    runtime->destroy_logical_region(ctx, lr_shared_range);

    // Ignore chunking for now

    writeStats();
}


void Mesh::writeStats() {

    coord_t gnump = nump;
    coord_t gnumz = numz;
    coord_t gnums = nums;

    LEGION_PRINT_ONCE(runtime, ctx, stdout, "--- Mesh Information ---\n");
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "Points:  %lld\n", gnump);
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "Zones:   %lld\n", gnumz);
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "Sides:   %lld\n", gnums);
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "Chunk size:   %d\n", chunksize);
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "------------------------\n");
}


void Mesh::write(
        const string& probname,
        const Future &f_cycle,
        const Future &f_time) {

    TaskLauncher launcher(TID_WRITE, TaskArgument(probname.c_str(), probname.size()+1));
    launcher.add_future(f_cycle);
    launcher.add_future(f_time);
    launcher.add_region_requirement(
        RegionRequirement(lrz, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lrz));
    launcher.add_field(0, FID_ZP);
    launcher.add_field(0, FID_ZE);
    launcher.add_field(0, FID_ZR);
    launcher.add_field(0, FID_ZNUMP);
    launcher.add_region_requirement(
        RegionRequirement(lrp, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lrp));
    launcher.add_field(1, FID_PX);
    launcher.add_region_requirement(
        RegionRequirement(lrs, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lrs));
    launcher.add_field(2, FID_MAPSP1);
    runtime->execute_task(ctx, launcher);
}


void Mesh::calcOwnershipParallel(
            Runtime *runtime,
            Context ctx,
            LogicalRegion lr_sides,
            LogicalPartition lp_sides,
            IndexPartition ip_private,
            IndexPartition ip_shared,
            IndexSpace is_piece) {
  const CalcOwnersArgs args(ip_private, ip_shared);
  IndexTaskLauncher launcher(TID_CALCOWNERS, is_piece, 
                            TaskArgument(&args, sizeof(args)), ArgumentMap());
  launcher.add_region_requirement(
      RegionRequirement(lp_sides, 0/*identity projection*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_sides));
  launcher.add_field(0/*index*/, FID_MAPSP1);
  launcher.add_field(0/*index*/, FID_MAPSP2);
  launcher.add_region_requirement(
      RegionRequirement(lp_sides, 0/*identity projection*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_sides));
  launcher.add_field(1/*index*/, FID_MAPSP1REG);
  launcher.add_field(1/*index*/, FID_MAPSP2REG);
  runtime->execute_index_space(ctx, launcher);
}


void Mesh::calcOwnersTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorWD<int> acc_mapsp1reg(regions[1], FID_MAPSP1REG);
    const AccessorWD<int> acc_mapsp2reg(regions[1], FID_MAPSP2REG);

    const CalcOwnersArgs *args = reinterpret_cast<const CalcOwnersArgs*>(task->args);
    const Domain private_domain = 
      runtime->get_index_space_domain(
          runtime->get_index_subspace(args->ip_private, task->index_point));
    const Domain shared_domain = 
      runtime->get_index_space_domain(
          runtime->get_index_subspace(args->ip_shared, task->index_point));

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (PointIterator itr(runtime, iss); itr(); itr++)
    {
      const Pointer p1 = acc_mapsp1[*itr];
      if (!private_domain.contains(p1))
      {
        assert(shared_domain.contains(p1));
        acc_mapsp1reg[*itr] = 1;
      }
      else
        acc_mapsp1reg[*itr] = 0;

      const Pointer p2 = acc_mapsp2[*itr];
      if (!private_domain.contains(p2))
      {
        assert(shared_domain.contains(p2));
        acc_mapsp2reg[*itr] = 1;
      }
      else
        acc_mapsp2reg[*itr] = 0;
    }
}


void Mesh::calcCtrsParallel(
            Runtime *runtime,
            Context ctx,
            LogicalRegion lr_sides,
            LogicalPartition lp_sides,
            LogicalRegion lr_zones,
            LogicalPartition lp_zones,
            LogicalRegion lr_points,
            LogicalPartition lp_points_private,
            LogicalPartition lp_points_shared,
            IndexSpace is_piece) {
  IndexTaskLauncher launcher(TID_CALCCTRS, is_piece, TaskArgument(), ArgumentMap());
  launcher.add_region_requirement(
      RegionRequirement(lp_sides, 0/*identity projection*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_sides));
  launcher.add_field(0/*index*/, FID_MAPSP1);
  launcher.add_field(0/*index*/, FID_MAPSP2);
  launcher.add_field(0/*index*/, FID_MAPSZ);
  launcher.add_field(0/*index*/, FID_MAPSP1REG);
  launcher.add_field(0/*index*/, FID_MAPSP2REG);
  launcher.add_region_requirement(
      RegionRequirement(lp_zones, 0/*identity projection*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_zones));
  launcher.add_field(1/*index*/, FID_ZNUMP);
  launcher.add_region_requirement(
      RegionRequirement(lp_points_private, 0/*identity*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_points));
  launcher.add_field(2/*index*/, FID_PX);
  launcher.add_region_requirement(
      RegionRequirement(lp_points_shared, 0/*identity*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_points));
  launcher.add_field(3/*index*/, FID_PX);
  launcher.add_region_requirement(
      RegionRequirement(lp_sides, 0/*identity*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_sides));
  launcher.add_field(4/*index*/, FID_EX);
  launcher.add_region_requirement(
      RegionRequirement(lp_zones, 0/*identity*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_zones));
  launcher.add_field(5/*index*/, FID_ZX);
  runtime->execute_index_space(ctx, launcher);
}


void Mesh::calcCtrsTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<int> acc_mapsp2reg(regions[0], FID_MAPSP2REG);
    const AccessorRO<int> acc_znump(regions[1], FID_ZNUMP);
    FieldID fid_px = task->regions[2].instance_fields[0];
    const AccessorRO<double2> acc_px[2] = {
        AccessorRO<double2>(regions[2], fid_px),
        AccessorRO<double2>(regions[3], fid_px)
    };
    FieldID fid_ex = task->regions[4].instance_fields[0];
    const AccessorWD<double2> acc_ex(regions[4], fid_ex);
    FieldID fid_zx = task->regions[5].instance_fields[0];
    const AccessorWD<double2> acc_zx(regions[5], fid_zx);

    const IndexSpace& isz = task->regions[1].region.get_index_space();
    for (PointIterator itr(runtime, isz); itr(); itr++)
      acc_zx[*itr] = double2(0., 0.);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (PointIterator itr(runtime, iss); itr(); itr++)
    {
        const Pointer p1 = acc_mapsp1[*itr];
        const int p1reg = acc_mapsp1reg[*itr];
        const Pointer p2 = acc_mapsp2[*itr];
        const int p2reg = acc_mapsp2reg[*itr];
        const Pointer z = acc_mapsz[*itr];
        const double2 px1 = acc_px[p1reg][p1];
        const double2 px2 = acc_px[p2reg][p2];
        const double2 ex  = 0.5 * (px1 + px2);
        acc_ex[*itr] = ex;
        const int n = acc_znump[z];
        acc_zx[z] += px1 / n;
    }
}


void Mesh::calcCtrsOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<int> acc_mapsp2reg(regions[0], FID_MAPSP2REG);
    const AccessorRO<int> acc_znump(regions[1], FID_ZNUMP);
    FieldID fid_px = task->regions[2].instance_fields[0];
    const AccessorRO<double2> acc_px[2] = {
        AccessorRO<double2>(regions[2], fid_px),
        AccessorRO<double2>(regions[3], fid_px)
    };
    FieldID fid_ex = task->regions[4].instance_fields[0];
    const AccessorWD<double2> acc_ex(regions[4], fid_ex);
    FieldID fid_zx = task->regions[5].instance_fields[0];
    const AccessorWD<double2> acc_zx(regions[5], fid_zx);

    const IndexSpace& isz = task->regions[1].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    #pragma omp parallel for
    for (coord_t z = rectz.lo[0]; z <= rectz.hi[0]; z++)
      acc_zx[z] = double2(0., 0.);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    #pragma omp parallel for
    for (coord_t s = rects.lo[0]; s <= rects.hi[0]; s++)
    {
        const Pointer p1 = acc_mapsp1[s];
        const int p1reg = acc_mapsp1reg[s];
        const Pointer p2 = acc_mapsp2[s];
        const int p2reg = acc_mapsp2reg[s];
        const Pointer z = acc_mapsz[s];
        const double2 px1 = acc_px[p1reg][p1];
        const double2 px2 = acc_px[p2reg][p2];
        const double2 ex  = 0.5 * (px1 + px2);
        acc_ex[s] = ex;
        const int n = acc_znump[z];
        SumOp<double2>::apply<false/*exclusive*/>(acc_zx[z], px1 / n);
    }
}


Future Mesh::calcVolsParallel(
            Runtime *runtime,
            Context ctx,
            LogicalRegion lr_sides,
            LogicalPartition lp_sides,
            LogicalRegion lr_zones,
            LogicalPartition lp_zones,
            LogicalRegion lr_points,
            LogicalPartition lp_points_private,
            LogicalPartition lp_points_shared,
            IndexSpace is_piece) {
  IndexTaskLauncher launcher(TID_CALCVOLS, is_piece, TaskArgument(), ArgumentMap());
  launcher.add_region_requirement(
      RegionRequirement(lp_sides, 0/*identity projection*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_sides));
  launcher.add_field(0/*index*/, FID_MAPSP1);
  launcher.add_field(0/*index*/, FID_MAPSP2);
  launcher.add_field(0/*index*/, FID_MAPSZ);
  launcher.add_field(0/*index*/, FID_MAPSP1REG);
  launcher.add_field(0/*index*/, FID_MAPSP2REG);
  launcher.add_region_requirement(
      RegionRequirement(lp_points_private, 0/*identity*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_points));
  launcher.add_field(1/*index*/, FID_PX);
  launcher.add_region_requirement(
      RegionRequirement(lp_points_shared, 0/*identity*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_points));
  launcher.add_field(2/*index*/, FID_PX);
  launcher.add_region_requirement(
      RegionRequirement(lp_zones, 0/*identity*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_zones));
  launcher.add_field(3/*index*/, FID_ZX);
  launcher.add_region_requirement(
      RegionRequirement(lp_sides, 0/*identity*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_sides));
  launcher.add_field(4/*index*/, FID_SAREA);
  launcher.add_field(4/*index*/, FID_SVOL);
  launcher.add_region_requirement(
      RegionRequirement(lp_zones, 0/*identity*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_zones));
  launcher.add_field(5/*index*/, FID_ZAREA);
  launcher.add_field(5/*index*/, FID_ZVOL);
  return runtime->execute_index_space(ctx, launcher, OPID_SUMINT);
}

int Mesh::calcVolsTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<int> acc_mapsp2reg(regions[0], FID_MAPSP2REG);
    FieldID fid_px = task->regions[1].instance_fields[0];
    const AccessorRO<double2> acc_px[2] = {
        AccessorRO<double2>(regions[1], fid_px),
        AccessorRO<double2>(regions[2], fid_px)
    };
    FieldID fid_zx = task->regions[3].instance_fields[0];
    const AccessorRO<double2> acc_zx(regions[3], fid_zx);
    FieldID fid_sarea = task->regions[4].instance_fields[0];
    FieldID fid_svol  = task->regions[4].instance_fields[1];
    const AccessorWD<double> acc_sarea(regions[4], fid_sarea);
    const AccessorWD<double> acc_svol(regions[4], fid_svol);
    FieldID fid_zarea = task->regions[5].instance_fields[0];
    FieldID fid_zvol  = task->regions[5].instance_fields[1];
    const AccessorWD<double> acc_zarea(regions[5], fid_zarea);
    const AccessorWD<double> acc_zvol(regions[5], fid_zvol);

    const IndexSpace& isz = task->regions[3].region.get_index_space();
    for (PointIterator itr(runtime, isz); itr(); itr++)
    {
        acc_zarea[*itr] = 0.;
        acc_zvol[*itr] = 0.;
    }

    const double third = 1. / 3.;
    int count = 0;
    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (PointIterator itr(runtime, iss); itr(); itr++)
    {
        const Pointer p1 = acc_mapsp1[*itr];
        const int p1reg = acc_mapsp1reg[*itr];
        const Pointer p2 = acc_mapsp2[*itr];
        const int p2reg = acc_mapsp2reg[*itr];
        const Pointer z = acc_mapsz[*itr];
        const double2 px1 = acc_px[p1reg][p1];
        const double2 px2 = acc_px[p2reg][p2];
        const double2 zx  = acc_zx[z];

        // compute side volumes, sum to zone
        const double sa = 0.5 * cross(px2 - px1, zx - px1);
        const double sv = third * sa * (px1.x + px2.x + zx.x);
        acc_sarea[*itr] = sa;
        acc_svol[*itr] = sv;
        acc_zarea[z] += sa;
        acc_zvol[z] += sv;

        // check for negative side volumes
        if (sv <= 0.) 
          count += 1;
    }

    return count;
}


int Mesh::calcVolsOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<int> acc_mapsp2reg(regions[0], FID_MAPSP2REG);
    FieldID fid_px = task->regions[1].instance_fields[0];
    const AccessorRO<double2> acc_px[2] = {
        AccessorRO<double2>(regions[1], fid_px),
        AccessorRO<double2>(regions[2], fid_px)
    };
    FieldID fid_zx = task->regions[3].instance_fields[0];
    const AccessorRO<double2> acc_zx(regions[3], fid_zx);
    FieldID fid_sarea = task->regions[4].instance_fields[0];
    FieldID fid_svol  = task->regions[4].instance_fields[1];
    const AccessorWD<double> acc_sarea(regions[4], fid_sarea);
    const AccessorWD<double> acc_svol(regions[4], fid_svol);
    FieldID fid_zarea = task->regions[5].instance_fields[0];
    FieldID fid_zvol  = task->regions[5].instance_fields[1];
    const AccessorWD<double> acc_zarea(regions[5], fid_zarea);
    const AccessorWD<double> acc_zvol(regions[5], fid_zvol);

    const IndexSpace& isz = task->regions[3].region.get_index_space();
    // This will assert if it isn't dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    #pragma omp parallel for
    for (coord_t z = rectz.lo[0]; z <= rectz.hi[0]; z++)
    {
        acc_zarea[z] = 0.;
        acc_zvol[z] = 0.;
    }

    const double third = 1. / 3.;
    int count = 0;
    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it isn't dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    #pragma omp parallel for
    for (coord_t s = rects.lo[0]; s <= rects.hi[0]; s++)
    {
        const Pointer p1 = acc_mapsp1[s];
        const int p1reg = acc_mapsp1reg[s];
        const Pointer p2 = acc_mapsp2[s];
        const int p2reg = acc_mapsp2reg[s];
        const Pointer z = acc_mapsz[s];
        const double2 px1 = acc_px[p1reg][p1];
        const double2 px2 = acc_px[p2reg][p2];
        const double2 zx  = acc_zx[z];

        // compute side volumes, sum to zone
        const double sa = 0.5 * cross(px2 - px1, zx - px1);
        const double sv = third * sa * (px1.x + px2.x + zx.x);
        acc_sarea[s] = sa;
        acc_svol[s] = sv;
        SumOp<double>::apply<false/*exclusive*/>(acc_zarea[z], sa);
        SumOp<double>::apply<false/*exclusive*/>(acc_zvol[z], sv);

        // check for negative side volumes
        if (sv <= 0.) 
          SumOp<int>::apply<false/*exclusive*/>(count, 1);
    }

    return count;
}


void Mesh::calcSideFracsParallel(
            Runtime *runtime,
            Context ctx,
            LogicalRegion lr_sides,
            LogicalPartition lp_sides,
            LogicalRegion lr_zones,
            LogicalPartition lp_zones,
            IndexSpace is_piece) {
  IndexTaskLauncher launcher(TID_CALCSIDEFRACS, is_piece, TaskArgument(), ArgumentMap());
  launcher.add_region_requirement(
      RegionRequirement(lp_sides, 0/*identity*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_sides));
  launcher.add_field(0/*index*/, FID_SAREA);
  launcher.add_field(0/*index*/, FID_MAPSZ);
  launcher.add_region_requirement(
      RegionRequirement(lp_zones, 0/*identity*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_zones));
  launcher.add_field(1/*index*/, FID_ZAREA);
  launcher.add_region_requirement(
      RegionRequirement(lp_sides, 0/*idenity*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_sides));
  launcher.add_field(2/*index*/, FID_SMF);
  runtime->execute_index_space(ctx, launcher);
}


void Mesh::calcSideFracsTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<double> acc_sarea(regions[0], FID_SAREA);
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<double> acc_zarea(regions[1], FID_ZAREA);
    const AccessorWD<double> acc_smf(regions[2], FID_SMF);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (PointIterator itr(runtime, iss); itr(); itr++)
    {
      const Pointer z = acc_mapsz[*itr];
      acc_smf[*itr] = acc_sarea[*itr] / acc_zarea[z];
    }
}


void Mesh::calcSurfVecsTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<double2> acc_ex(regions[0], FID_EXP);
    const AccessorRO<double2> acc_zx(regions[1], FID_ZXP);
    const AccessorWD<double2> acc_ssurf(regions[2], FID_SSURFP);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (PointIterator itr(runtime, iss); itr(); itr++)
    {
        const Pointer z = acc_mapsz[*itr];
        const double2 ex = acc_ex[*itr];
        const double2 zx = acc_zx[z];
        const double2 ss = rotateCCW(ex - zx);
        acc_ssurf[*itr] = ss;
    }
}


void Mesh::calcSurfVecsOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<double2> acc_ex(regions[0], FID_EXP);
    const AccessorRO<double2> acc_zx(regions[1], FID_ZXP);
    const AccessorWD<double2> acc_ssurf(regions[2], FID_SSURFP);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    #pragma omp parallel for
    for (coord_t s = rects.lo[0]; s <= rects.hi[0]; s++)
    {
        const Pointer z = acc_mapsz[s];
        const double2 ex = acc_ex[s];
        const double2 zx = acc_zx[z];
        const double2 ss = rotateCCW(ex - zx);
        acc_ssurf[s] = ss;
    }
}


void Mesh::calcEdgeLenTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<int> acc_mapsp2reg(regions[0], FID_MAPSP2REG);
    const AccessorRO<double2> acc_px[2] = {
        AccessorRO<double2>(regions[1], FID_PXP),
        AccessorRO<double2>(regions[2], FID_PXP)
    };
    const AccessorWD<double> acc_elen(regions[3], FID_ELEN);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (PointIterator itr(runtime, iss); itr(); itr++)
    {
        const Pointer p1 = acc_mapsp1[*itr];
        const int p1reg = acc_mapsp1reg[*itr];
        const Pointer p2 = acc_mapsp2[*itr];
        const int p2reg = acc_mapsp2reg[*itr];
        const double2 px1 = acc_px[p1reg][p1];
        const double2 px2 = acc_px[p2reg][p2];

        const double elen = length(px2 - px1);
        acc_elen[*itr] = elen;
    }
}


void Mesh::calcEdgeLenOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<int> acc_mapsp2reg(regions[0], FID_MAPSP2REG);
    const AccessorRO<double2> acc_px[2] = {
        AccessorRO<double2>(regions[1], FID_PXP),
        AccessorRO<double2>(regions[2], FID_PXP)
    };
    const AccessorWD<double> acc_elen(regions[3], FID_ELEN);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    #pragma omp parallel for
    for (coord_t s = rects.lo[0]; s <= rects.hi[0]; s++)
    {
        const Pointer p1 = acc_mapsp1[s];
        const int p1reg = acc_mapsp1reg[s];
        const Pointer p2 = acc_mapsp2[s];
        const int p2reg = acc_mapsp2reg[s];
        const double2 px1 = acc_px[p1reg][p1];
        const double2 px2 = acc_px[p2reg][p2];

        const double elen = length(px2 - px1);
        acc_elen[s] = elen;
    }
}


void Mesh::calcCharLenTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<double> acc_elen(regions[0], FID_ELEN);
    const AccessorRO<double> acc_sarea(regions[0], FID_SAREAP);
    const AccessorRO<int> acc_znump(regions[1], FID_ZNUMP);
    const AccessorWD<double> acc_zdl(regions[2], FID_ZDL);

    const IndexSpace& isz = task->regions[1].region.get_index_space();
    for (PointIterator itr(runtime, isz); itr(); itr++)
        acc_zdl[*itr] = 1.e99;
    
    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (PointIterator itr(runtime, iss); itr(); itr++)
    {
        const Pointer z = acc_mapsz[*itr];
        const double area = acc_sarea[*itr];
        const double base = acc_elen[*itr];
        const double zdl = acc_zdl[z];
        const int np = acc_znump[z];
        const double fac = (np == 3 ? 3. : 4.);
        const double sdl = fac * area / base;
        const double zdl2 = min(zdl, sdl);
        acc_zdl[z] = zdl2;
    }
}


void Mesh::calcCharLenOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<double> acc_elen(regions[0], FID_ELEN);
    const AccessorRO<double> acc_sarea(regions[0], FID_SAREAP);
    const AccessorRO<int> acc_znump(regions[1], FID_ZNUMP);
    const AccessorWD<double> acc_zdl(regions[2], FID_ZDL);

    const IndexSpace& isz = task->regions[1].region.get_index_space();
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    #pragma omp parallel for
    for (coord_t z = rectz.lo[0]; z <= rectz.hi[0]; z++)
        acc_zdl[z] = 1.e99;
    
    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it isn't dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    #pragma omp parallel for
    for (coord_t s = rects.lo[0]; s <= rects.hi[0]; s++)
    {
        const Pointer z = acc_mapsz[s];
        const double area = acc_sarea[s];
        const double base = acc_elen[s];
        const int np = acc_znump[z];
        const double fac = (np == 3 ? 3. : 4.);
        const double sdl = fac * area / base;
        MinOp<double>::apply<false/*exclusive*/>(acc_zdl[z], sdl);
    }
}


#if 0
void Mesh::calcCtrs(
        const double2* px,
        double2* ex,
        double2* zx,
        const int sfirst,
        const int slast) {

    int zfirst = mapsz[sfirst];
    int zlast = (slast < nums ? mapsz[slast] : numz);
    fill(&zx[zfirst], &zx[zlast], double2(0., 0.));

    for (int s = sfirst; s < slast; ++s) {
        int p1 = mapsp1[s];
        int p2 = mapsp2[s];
        int z = mapsz[s];
        ex[s] = 0.5 * (px[p1] + px[p2]);
        zx[z] += px[p1];
    }

    for (int z = zfirst; z < zlast; ++z) {
        zx[z] /= (double) znump[z];
    }

}


int Mesh::calcVols(
        const double2* px,
        const double2* zx,
        double* sarea,
        double* svol,
        double* zarea,
        double* zvol,
        const int sfirst,
        const int slast) {

    int zfirst = mapsz[sfirst];
    int zlast = (slast < nums ? mapsz[slast] : numz);
    fill(&zvol[zfirst], &zvol[zlast], 0.);
    fill(&zarea[zfirst], &zarea[zlast], 0.);

    const double third = 1. / 3.;
    int count = 0;
    for (int s = sfirst; s < slast; ++s) {
        int p1 = mapsp1[s];
        int p2 = mapsp2[s];
        int z = mapsz[s];

        // compute side volumes, sum to zone
        double sa = 0.5 * cross(px[p2] - px[p1], zx[z] - px[p1]);
        double sv = third * sa * (px[p1].x + px[p2].x + zx[z].x);
        sarea[s] = sa;
        svol[s] = sv;
        zarea[z] += sa;
        zvol[z] += sv;

        // check for negative side volumes
        if (sv <= 0.) count += 1;

    } // for s

    return count;
}
#endif


void Mesh::checkBadSides(int cycle, Future f, Predicate pred) {

    // We launch a task to check this to avoid blocking
    // the top-level task
    TaskLauncher launcher(TID_CHECKBADSIDES, 
        TaskArgument(&cycle, sizeof(cycle)), pred);
    launcher.add_future(f);
    runtime->execute_task(ctx, launcher);

}


#if 0
void Mesh::calcSideFracs(
        const double* sarea,
        const double* zarea,
        double* smf,
        const int sfirst,
        const int slast) {

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int z = mapsz[s];
        smf[s] = sarea[s] / zarea[z];
    }
}
#endif


void Mesh::computeRangesParallel(
            const int numpcs,
            Runtime *runtime,
            Context ctx,
            LogicalRegion lr_all_range,
            LogicalRegion lr_private_range,
            LogicalPartition lp_private_range,
            LogicalRegion lr_shared_range,
            LogicalPartition lp_shared_range,
            IndexPartition ip_private,
            IndexPartition ip_shared,
            IndexSpace is_piece)
{
  // First we do two index space launches to compute the counts of
  // the number of points in each subregion
  {
    IndexTaskLauncher launcher(TID_COUNTPOINTS, is_piece,
        TaskArgument(&ip_private, sizeof(ip_private)), ArgumentMap());
    launcher.add_region_requirement(RegionRequirement(lp_private_range,
          0/*identity projection*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_private_range));
    launcher.add_field(0/*index*/, FID_COUNT);
    runtime->execute_index_space(ctx, launcher);
  }
  {
    IndexTaskLauncher launcher(TID_COUNTPOINTS, is_piece,
        TaskArgument(&ip_shared, sizeof(ip_shared)), ArgumentMap());
    launcher.add_region_requirement(RegionRequirement(lp_shared_range,
          0/*identity projection*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_shared_range));
    launcher.add_field(0/*index*/, FID_COUNT);
    runtime->execute_index_space(ctx, launcher);
  }
  // Then we do a single task launch to compute the ranges
  TaskLauncher launcher(TID_CALCRANGES, TaskArgument());
  launcher.add_region_requirement(
      RegionRequirement(lr_all_range, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_all_range));
  launcher.add_field(0/*index*/, FID_RANGE);
  launcher.add_region_requirement(
      RegionRequirement(lr_private_range, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_private_range));
  launcher.add_field(1/*index*/, FID_RANGE);
  launcher.add_region_requirement(
      RegionRequirement(lr_shared_range, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_shared_range));
  launcher.add_field(2/*index*/, FID_RANGE);
  launcher.add_region_requirement(
      RegionRequirement(lr_private_range, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_private_range));
  launcher.add_field(3/*index*/, FID_COUNT);
  launcher.add_region_requirement(
      RegionRequirement(lr_shared_range, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_shared_range));
  launcher.add_field(4/*index*/, FID_COUNT);
  runtime->execute_task(ctx, launcher);
}


void Mesh::compactPointsParallel(
            const int numpcs,
            Runtime *runtime,
            Context ctx,
            LogicalRegion lr_temp_points,
            LogicalPartition lp_temp_points,
            LogicalRegion lr_points,
            LogicalPartition lp_points,
            IndexSpace is_piece) {
  IndexTaskLauncher launcher(TID_COMPACTPOINTS, is_piece,
                              TaskArgument(), ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(lp_points,
        0/*identity projection*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_points));
  launcher.add_field(0/*index*/, FID_PX);
  launcher.add_region_requirement(RegionRequirement(lp_temp_points,
        0/*identity projection*/, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr_temp_points));
  launcher.add_field(1/*index*/, FID_PX);
  launcher.add_region_requirement(RegionRequirement(lp_temp_points,
        0/*identity projection*/, LEGION_WRITE_DISCARD, LEGION_EXCLUSIVE, lr_temp_points));
  launcher.add_field(2/*index*/, FID_MAPLOAD2DENSE);

  runtime->execute_index_space(ctx, launcher);
}


void Mesh::countPointsTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    IndexPartition ip = *(const IndexPartition*)task->args;
    // Get the subregion
    IndexSpace is = runtime->get_index_subspace(ip, task->index_point);
    Domain dom = runtime->get_index_space_domain(is);
    // Write out the count of the number of points
    const AccessorWD<coord_t> acc(regions[0], FID_COUNT);
    acc[task->index_point] = dom.get_volume();
}


void Mesh::calcRangesTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    coord_t current = 0;
    // Compute the private ranges first  
    const AccessorRO<coord_t> priv_count(regions[3], FID_COUNT);
    const AccessorWD<Rect<1> > priv_range(regions[1], FID_RANGE);
    IndexSpace is_priv = task->regions[3].region.get_index_space();
    for (PointIterator itr(runtime, is_priv); itr(); itr++)
    {
      const coord_t count = priv_count[*itr];
      priv_range[*itr] = Rect<1>(current, current + count - 1);
      current += count;
    }
    // Update the all private range
    const AccessorWD<Rect<1> > all_range(regions[0], FID_RANGE);
    const coord_t shared_start = current;
    all_range[Pointer(0)] = Rect<1>(0, shared_start-1);
    // Now we can do the shared ranges
    const AccessorRO<coord_t> shr_count(regions[4], FID_COUNT);
    const AccessorWD<Rect<1> > shr_range(regions[2], FID_RANGE);
    IndexSpace is_shr = task->regions[4].region.get_index_space();
    for (PointIterator itr(runtime, is_shr); itr(); itr++)
    {
      const coord_t count = shr_count[*itr];
      shr_range[*itr] = Rect<1>(current, current + count - 1);
      current += count;
    }
    // Update the all shared range
    all_range[Pointer(1)] = Rect<1>(shared_start, current-1);
}


void Mesh::compactPointsTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    IndexSpace is_dst = task->regions[0].region.get_index_space();
    IndexSpace is_src = task->regions[1].region.get_index_space();
    assert(runtime->get_index_space_domain(is_src).get_volume() ==
            runtime->get_index_space_domain(is_dst).get_volume());
    PointIterator itr_src(runtime, is_src);
    PointIterator itr_dst(runtime, is_dst);
    const AccessorWD<double2> acc_dst(regions[0], FID_PX);
    const AccessorRO<double2> acc_src(regions[1], FID_PX);
    const AccessorWD<Pointer> acc_ptr(regions[2], FID_MAPLOAD2DENSE);
    for ( ; itr_src() && itr_dst(); itr_src++, itr_dst++)
    {
#ifdef PRECOMPACTED_RECT_POINTS
      // the whole point of this code is that "compaction" should be an
      //  identity map
      assert(*itr_src == *itr_dst);
#endif
      acc_dst[*itr_dst] = acc_src[*itr_src];
      acc_ptr[*itr_src] = *itr_dst;
    }
}


void Mesh::checkBadSidesTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    assert(task->futures.size() == 1);
    const int numsbad = task->futures[0].get_result<int>();
    // if there were negative side volumes, error exit
    if (numsbad > 0) {
        const int cycle = *reinterpret_cast<const int*>(task->args);
        if (cycle >= 0)
            cerr << "Error: " << numsbad << " negative side volumes on cycle " 
                 << cycle << endl;
        else
            cerr << "Error: " << numsbad << " negative side volumes " 
                 << "during initialization" << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }
}


void Mesh::tempGatherTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_src(regions[0], task->regions[0].instance_fields[0]);
    const AccessorWD<Pointer> acc_dst(regions[1], task->regions[1].instance_fields[0]);
    const AccessorRO<Pointer> acc_idx(regions[2], task->regions[2].instance_fields[0]);

    const IndexSpace iss = task->regions[1].region.get_index_space(); 
    for (PointIterator itr(runtime, iss); itr(); itr++)
    {
      // For the output figure out where we want to do the gather from
      const Pointer p = acc_idx[*itr];
      acc_dst[*itr] = acc_src[p];
    }
}


void Mesh::writeTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
  const AccessorRO<double> acc_zp(regions[0], FID_ZP);
  const AccessorRO<double> acc_ze(regions[0], FID_ZE);
  const AccessorRO<double> acc_zr(regions[0], FID_ZR);
  const AccessorRO<int> acc_znump(regions[0], FID_ZNUMP);
  const Rect<1> zone_bounds = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  const double *zp = acc_zp.ptr(zone_bounds);
  const double *ze = acc_ze.ptr(zone_bounds);
  const double *zr = acc_zr.ptr(zone_bounds);
  const int *znump = acc_znump.ptr(zone_bounds);

  const AccessorRO<double2> acc_px(regions[1], FID_PX);
  const Rect<1> point_bounds = runtime->get_index_space_domain(ctx,
      task->regions[1].region.get_index_space());
  const double2 *px = acc_px.ptr(point_bounds);

  const AccessorRO<Pointer> acc_mapsp1(regions[2], FID_MAPSP1);
  const Rect<1> side_bounds = runtime->get_index_space_domain(ctx,
      task->regions[2].region.get_index_space());
  const Pointer *mapsp1 = acc_mapsp1.ptr(side_bounds);

  std::string probname((const char*)task->args);
  WriteXY wxy;
  wxy.write(probname, zr, ze, zp, zone_bounds.volume());
  ExportGold egold;
  egold.write(probname, task->futures[0].get_result<int>(),
      task->futures[1].get_result<double>(), 
      zr, ze, zp, znump, zone_bounds.volume(), 
      px, point_bounds.volume(), mapsp1);
}
