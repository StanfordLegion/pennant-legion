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

#include "legion.h"

#include "MyLegion.hh"
#include "Vec2.hh"
#include "Memory.hh"
#include "InputFile.hh"
#include "GenMesh.hh"
#include "WriteXY.hh"
#include "ExportGold.hh"

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
      TaskVariantRegistrar registrar(TID_CALCVOLS, "CPU calcvols");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<int, Mesh::calcVolsTask>(registrar, "calcvols");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCSURFVECS, "CPU calcsurfvecs");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcSurfVecsTask>(registrar, "calcsurfvecs");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCEDGELEN, "CPU calcedgelen");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcEdgeLenTask>(registrar, "calcedgelen");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCCHARLEN, "CPU calccharlen");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcCharLenTask>(registrar, "calccharlen");
    }

    Runtime::register_reduction_op<SumOp<int> >(
            OPID_SUMINT);
    Runtime::register_reduction_op<SumOp<double> >(
            OPID_SUMDBL);
    Runtime::register_reduction_op<SumOp<double2> >(
            OPID_SUMDBL2);
    Runtime::register_reduction_op<MinOp<double> >(
            OPID_MINDBL);
}
}; // namespace


template <>
void atomicAdd(int& lhs, const int& rhs) {
    __sync_add_and_fetch(&lhs, rhs);
}


template <>
void atomicAdd(double& lhs, const double& rhs) {
    long long *target = (long long *)&lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}


template <>
void atomicAdd(double2& lhs, const double2& rhs) {
    atomicAdd(lhs.x, rhs.x);
    atomicAdd(lhs.y, rhs.y);
}


template <>
void atomicMin(double& lhs, const double& rhs) {
    long long *target = (long long *)&lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = min(oldval.as_float, rhs);
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}


template <>
const int SumOp<int>::identity = 0;
template <>
const double SumOp<double>::identity = 0.;
template <>
const double2 SumOp<double2>::identity = double2(0., 0.);
template <>
const double MinOp<double>::identity = 1.e99;


Mesh::Mesh(
        const InputFile* inp,
        const int numpcsa,
        const bool parallel,
        Context ctxa,
        Runtime* runtimea)
        : gmesh(NULL),  wxy(NULL), egold(NULL),
          numpcs(numpcsa), ctx(ctxa), runtime(runtimea) {

    chunksize = inp->getInt("chunksize", 0);
    subregion = inp->getDoubleList("subregion", vector<double>());
    if (subregion.size() != 0 && subregion.size() != 4) {
        cerr << "Error:  subregion must have 4 entries" << endl;
        exit(1);
    }

    gmesh = new GenMesh(inp);
    wxy = new WriteXY(this);
    egold = new ExportGold(this);

    if (parallel)
        initParallel();
    else
        init();
}


Mesh::~Mesh() {
    delete gmesh;
    delete wxy;
    delete egold;
}


void Mesh::init() {

    // generate mesh
    vector<double2> nodepos;
    vector<int> cellstart, cellsize, cellnodes;
    vector<int> cellcolors;
    gmesh->generate(numpcs, nodepos, nodecolors, nodemcolors,
            cellstart, cellsize, cellnodes, cellcolors);

    nump = nodepos.size();
    numz = cellstart.size();
    nums = cellnodes.size();
    numc = nums;

    // copy cell sizes to mesh
    znump = alloc<int>(numz);
    copy(cellsize.begin(), cellsize.end(), znump);

    // populate maps:
    // use the cell* arrays to populate the side maps
    initSides(cellstart, cellsize, cellnodes);
    // release memory from cell* arrays
    cellstart.resize(0);
    cellsize.resize(0);
    cellnodes.resize(0);

    // populate chunk information
    initChunks();

    // write mesh statistics
    writeStats();

    // allocate remaining arrays
    px = alloc<double2>(nump);
    ex = alloc<double2>(nums);
    zx = alloc<double2>(numz);
    sarea = alloc<double>(nums);
    svol = alloc<double>(nums);
    zarea = alloc<double>(numz);
    zvol = alloc<double>(numz);
    smf = alloc<double>(nums);

    // do a few initial calculations
    for (int pch = 0; pch < numpch; ++pch) {
        int pfirst = pchpfirst[pch];
        int plast = pchplast[pch];
        // copy nodepos into px, distributed across threads
        for (int p = pfirst; p < plast; ++p)
            px[p] = nodepos[p];

    }

    numsbad = 0;
    for (int sch = 0; sch < numsch; ++sch) {
        int sfirst = schsfirst[sch];
        int slast = schslast[sch];
        calcCtrs(px, ex, zx, sfirst, slast);
        calcVols(px, zx, sarea, svol, zarea, zvol, sfirst, slast);
        calcSideFracs(sarea, zarea, smf, sfirst, slast);
    }
    // check for negative volumes on initialization
    if (numsbad > 0) {
        cerr << "Error: " << numsbad << " negative side volumes" << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

    // create index spaces and fields for points, zones, sides
    IndexSpace isp = runtime->create_index_space(ctx, nump);
    FieldSpace fsp = runtime->create_field_space(ctx);
    FieldAllocator fap = runtime->create_field_allocator(ctx, fsp);
    fap.allocate_field(sizeof(double2), FID_PX);
    fap.allocate_field(sizeof(double2), FID_PXP);
    fap.allocate_field(sizeof(double2), FID_PX0);
    fap.allocate_field(sizeof(double2), FID_PU);
    fap.allocate_field(sizeof(double2), FID_PU0);
    fap.allocate_field(sizeof(double), FID_PMASWT);
    fap.allocate_field(sizeof(double2), FID_PF);
    fap.allocate_field(sizeof(double2), FID_PAP);
    lrp = runtime->create_logical_region(ctx, isp, fsp);
    runtime->attach_name(lrp, "lrp");

    IndexSpace isz = runtime->create_index_space(ctx, numz);
    FieldSpace fsz = runtime->create_field_space(ctx);
    FieldAllocator faz = runtime->create_field_allocator(ctx, fsz);
    faz.allocate_field(sizeof(int), FID_ZNUMP);
    faz.allocate_field(sizeof(double2), FID_ZX);
    faz.allocate_field(sizeof(double2), FID_ZXP);
    faz.allocate_field(sizeof(double), FID_ZAREA);
    faz.allocate_field(sizeof(double), FID_ZVOL);
    faz.allocate_field(sizeof(double), FID_ZAREAP);
    faz.allocate_field(sizeof(double), FID_ZVOLP);
    faz.allocate_field(sizeof(double), FID_ZVOL0);
    faz.allocate_field(sizeof(double), FID_ZDL);
    faz.allocate_field(sizeof(double), FID_ZM);
    faz.allocate_field(sizeof(double), FID_ZR);
    faz.allocate_field(sizeof(double), FID_ZRP);
    faz.allocate_field(sizeof(double), FID_ZE);
    faz.allocate_field(sizeof(double), FID_ZETOT);
    faz.allocate_field(sizeof(double), FID_ZW);
    faz.allocate_field(sizeof(double), FID_ZWRATE);
    faz.allocate_field(sizeof(double), FID_ZP);
    faz.allocate_field(sizeof(double), FID_ZSS);
    faz.allocate_field(sizeof(double), FID_ZDU);
    faz.allocate_field(sizeof(double2), FID_ZUC);
    faz.allocate_field(sizeof(double), FID_ZTMP);
    lrz = runtime->create_logical_region(ctx, isz, fsz);
    runtime->attach_name(lrz, "lrz");

    IndexSpace iss = runtime->create_index_space(ctx, nums);
    FieldSpace fss = runtime->create_field_space(ctx);
    FieldAllocator fas = runtime->create_field_allocator(ctx, fss);
    fas.allocate_field(sizeof(Pointer), FID_MAPSP1);
    fas.allocate_field(sizeof(Pointer), FID_MAPSP2);
    fas.allocate_field(sizeof(Pointer), FID_MAPSZ);
    fas.allocate_field(sizeof(Pointer), FID_MAPSS3);
    fas.allocate_field(sizeof(Pointer), FID_MAPSS4);
    fas.allocate_field(sizeof(int), FID_MAPSP1REG);
    fas.allocate_field(sizeof(int), FID_MAPSP2REG);
    fas.allocate_field(sizeof(double2), FID_EX);
    fas.allocate_field(sizeof(double2), FID_EXP);
    fas.allocate_field(sizeof(double), FID_SAREA);
    fas.allocate_field(sizeof(double), FID_SVOL);
    fas.allocate_field(sizeof(double), FID_SAREAP);
    fas.allocate_field(sizeof(double), FID_SVOLP);
    fas.allocate_field(sizeof(double2), FID_SSURFP);
    fas.allocate_field(sizeof(double), FID_ELEN);
    fas.allocate_field(sizeof(double), FID_SMF);
    fas.allocate_field(sizeof(double2), FID_SFP);
    fas.allocate_field(sizeof(double2), FID_SFQ);
    fas.allocate_field(sizeof(double2), FID_SFT);
    fas.allocate_field(sizeof(double), FID_CAREA);
    fas.allocate_field(sizeof(double), FID_CEVOL);
    fas.allocate_field(sizeof(double), FID_CDU);
    fas.allocate_field(sizeof(double), FID_CDIV);
    fas.allocate_field(sizeof(double), FID_CCOS);
    fas.allocate_field(sizeof(double2), FID_CQE1);
    fas.allocate_field(sizeof(double2), FID_CQE2);
    fas.allocate_field(sizeof(double), FID_CRMU);
    fas.allocate_field(sizeof(double), FID_CW);
    lrs = runtime->create_logical_region(ctx, iss, fss);
    runtime->attach_name(lrs, "lrs");

    // create index spaces and fields for global vars
    IndexSpace isglb = runtime->create_index_space(ctx, 1);
    FieldSpace fsglb = runtime->create_field_space(ctx);
    FieldAllocator faglb = runtime->create_field_allocator(ctx, fsglb);
    faglb.allocate_field(sizeof(int), FID_NUMSBAD);
    faglb.allocate_field(sizeof(double), FID_DTREC);
    lrglb = runtime->create_logical_region(ctx, isglb, fsglb);
    runtime->attach_name(lrglb, "lrglb");

    // create domain over pieces
    Rect<1> task_rect(Point<1>(0), Point<1>(numpcs-1));
    dompc  = Domain(task_rect);
#if 0 
    IndexSpace ispc = runtime->create_index_space(ctx, dompc);
    dompc = runtime->get_index_space_domain(ctx, ispc);
#endif 
    // create zone and side partitions
    Coloring colorz, colors;
    int z0 = 0;
    while (z0 < numz)
    {
        int s0 = cellstart[z0];
        int c = cellcolors[z0];
        int z1 = z0 + 1;
        // find range with [z0, z1) all the same color
        while (z1 < numz && cellcolors[z1] == c) ++z1;
        int s1 = (z1 < numz ? cellstart[z1] : nums);
        colorz[c].ranges.insert(pair<int, int>(z0, z1 - 1));
        colors[c].ranges.insert(pair<int, int>(s0, s1 - 1));
        z0 = z1;
    }
    IndexPartition ipz = runtime->create_index_partition(
                ctx, isz, colorz, true);
    lpz = runtime->get_logical_partition(ctx, lrz, ipz);
    IndexPartition ips = runtime->create_index_partition(
                ctx, iss, colors, true);
    lps = runtime->get_logical_partition(ctx, lrs, ips);

    // create point partitions
    Coloring colorpall, colorpprv, colorpshr, colorpmstr;
    // force all colors to exist, even if they might be empty
    colorpall[0];
    colorpall[1];
    for (int c = 0; c < numpcs; ++c) {
        colorpprv[c];
        colorpmstr[c];
        colorpshr[c];
    }
    int p0 = 0;
    while (p0 < nump)
    {
        int c = nodecolors[p0];
        int p1 = p0 + 1;
        if (c != MULTICOLOR) {
            // find range with [p0, p1) all the same color
            while (p1 < nump && nodecolors[p1] == c) ++p1;
            colorpall[0].ranges.insert(pair<int, int>(p0, p1 - 1));
            colorpprv[c].ranges.insert(pair<int, int>(p0, p1 - 1));
        }
        else {
            // insert p0 by itself
            colorpall[1].points.insert(p0);
            vector<int>& pmc = nodemcolors[p0];
            colorpmstr[pmc[0]].points.insert(p0);
            for (int i = 0; i < pmc.size(); ++i)
                colorpshr[pmc[i]].points.insert(p0);
        }
        p0 = p1;
    }
    IndexPartition ippall = runtime->create_index_partition(
                ctx, isp, colorpall, true);
    lppall = runtime->get_logical_partition(ctx, lrp, ippall);
    IndexSpace ispprv = runtime->get_index_subspace(ctx, ippall, 0);
    IndexSpace ispshr = runtime->get_index_subspace(ctx, ippall, 1);

    IndexPartition ippprv = runtime->create_index_partition(
                ctx, ispprv, colorpprv, true);
    lppprv = runtime->get_logical_partition_by_tree(
            ctx, ippprv, fsp, lrp.get_tree_id());
    IndexPartition ippmstr = runtime->create_index_partition(
                ctx, ispshr, colorpmstr, true);
    lppmstr = runtime->get_logical_partition_by_tree(
            ctx, ippmstr, fsp, lrp.get_tree_id());
    IndexPartition ippshr = runtime->create_index_partition(
                ctx, ispshr, colorpshr, false);
    lppshr = runtime->get_logical_partition_by_tree(
            ctx, ippshr, fsp, lrp.get_tree_id());

    vector<Pointer> lgmapsp1(&mapsp1[0], &mapsp1[nums]);
    vector<Pointer> lgmapsp2(&mapsp2[0], &mapsp2[nums]);
    vector<Pointer> lgmapsz (&mapsz [0], &mapsz [nums]);
    vector<Pointer> lgmapss3(&mapss3[0], &mapss3[nums]);
    vector<Pointer> lgmapss4(&mapss4[0], &mapss4[nums]);

    vector<int> lgmapsp1reg(nums), lgmapsp2reg(nums);
    for (int s = 0; s < nums; ++s) {
        lgmapsp1reg[s] = (nodecolors[mapsp1[s]] == MULTICOLOR);
        lgmapsp2reg[s] = (nodecolors[mapsp2[s]] == MULTICOLOR);
    }

    setField(lrs, FID_MAPSP1, &lgmapsp1[0], nums);
    setField(lrs, FID_MAPSP2, &lgmapsp2[0], nums);
    setField(lrs, FID_MAPSZ,  &lgmapsz[0],  nums);
    setField(lrs, FID_MAPSS3, &lgmapss3[0], nums);
    setField(lrs, FID_MAPSS4, &lgmapss4[0], nums);
    setField(lrs, FID_MAPSP1REG, &lgmapsp1reg[0], nums);
    setField(lrs, FID_MAPSP2REG, &lgmapsp2reg[0], nums);

    setField(lrp, FID_PX, px, nump);
    setField(lrz, FID_ZVOL, zvol, numz);
    setField(lrz, FID_ZNUMP, znump, numz);
    setField(lrs, FID_SMF, smf, nums);

    setField(lrglb, FID_NUMSBAD, &numsbad, 1);

}


void Mesh::initParallel() {
    // Create a space for the number of pieces that we will have
    IndexSpace piece_is = runtime->create_index_space(ctx, Rect<1>(0, numpcs-1));
    IndexPartition piece_ip = runtime->create_equal_partition(ctx, piece_is, piece_is);
    // Create point index space and field spaces
    IndexSpace isp = runtime->create_index_space(ctx, 
                                      Rect<1>(0,gmesh->calcNumPoints(numpcs)-1));
    FieldSpace fsp = runtime->create_field_space(ctx);
    {
      FieldAllocator fap = runtime->create_field_allocator(ctx, fsp);
      fap.allocate_field(sizeof(double2), FID_PX);
      fap.allocate_field(sizeof(double2), FID_PXP);
      fap.allocate_field(sizeof(double2), FID_PX0);
      fap.allocate_field(sizeof(double2), FID_PU);
      fap.allocate_field(sizeof(double2), FID_PU0);
      fap.allocate_field(sizeof(double), FID_PMASWT);
      fap.allocate_field(sizeof(double2), FID_PF);
      fap.allocate_field(sizeof(double2), FID_PAP);
    }
    // load fields into temp points with equal partition
    LogicalRegion temp_points = runtime->create_logical_region(ctx, isp, fsp);
    IndexPartition points_equal = runtime->create_equal_partition(ctx, isp, piece_is);
    gmesh->generatePointsParallel(numpcs, runtime, ctx, temp_points, 
                            runtime->get_logical_partition(temp_points, points_equal));
    // equal partition zones
    IndexSpace isz = runtime->create_index_space(ctx, 
                        Rect<1>(0, gmesh->calcNumZones(numpcs)-1));
    FieldSpace fsz = runtime->create_field_space(ctx);
    {
      FieldAllocator faz = runtime->create_field_allocator(ctx, fsz);
      faz.allocate_field(sizeof(int), FID_ZNUMP);
      faz.allocate_field(sizeof(double2), FID_ZX);
      faz.allocate_field(sizeof(double2), FID_ZXP);
      faz.allocate_field(sizeof(double), FID_ZAREA);
      faz.allocate_field(sizeof(double), FID_ZVOL);
      faz.allocate_field(sizeof(double), FID_ZAREAP);
      faz.allocate_field(sizeof(double), FID_ZVOLP);
      faz.allocate_field(sizeof(double), FID_ZVOL0);
      faz.allocate_field(sizeof(double), FID_ZDL);
      faz.allocate_field(sizeof(double), FID_ZM);
      faz.allocate_field(sizeof(double), FID_ZR);
      faz.allocate_field(sizeof(double), FID_ZRP);
      faz.allocate_field(sizeof(double), FID_ZE);
      faz.allocate_field(sizeof(double), FID_ZETOT);
      faz.allocate_field(sizeof(double), FID_ZW);
      faz.allocate_field(sizeof(double), FID_ZWRATE);
      faz.allocate_field(sizeof(double), FID_ZP);
      faz.allocate_field(sizeof(double), FID_ZSS);
      faz.allocate_field(sizeof(double), FID_ZDU);
      faz.allocate_field(sizeof(double2), FID_ZUC);
      faz.allocate_field(sizeof(double), FID_ZTMP);
    }
    lrz = runtime->create_logical_region(ctx, isz, fsz);
    runtime->attach_name(lrz, "lrz");
    IndexPartition zones_equal = runtime->create_equal_partition(ctx, isz, piece_is);
    lpz = runtime->get_logical_partition(ctx, lrz, zones_equal);
    // Create temp side logical region
    IndexSpace iss = runtime->create_index_space(ctx, 
                          Rect<1>(0, gmesh->calcNumSides(numpcs)-1));
    FieldSpace fss = runtime->create_field_space(ctx);
    {
      FieldAllocator fas = runtime->create_field_allocator(ctx, fss);
      fas.allocate_field(sizeof(Pointer), FID_MAPSP1);
      fas.allocate_field(sizeof(Pointer), FID_MAPSP2);
      fas.allocate_field(sizeof(Pointer), FID_MAPSZ);
      fas.allocate_field(sizeof(Pointer), FID_MAPSS3);
      fas.allocate_field(sizeof(Pointer), FID_MAPSS4);
      fas.allocate_field(sizeof(int), FID_MAPSP1REG);
      fas.allocate_field(sizeof(int), FID_MAPSP2REG);
      fas.allocate_field(sizeof(double2), FID_EX);
      fas.allocate_field(sizeof(double2), FID_EXP);
      fas.allocate_field(sizeof(double), FID_SAREA);
      fas.allocate_field(sizeof(double), FID_SVOL);
      fas.allocate_field(sizeof(double), FID_SAREAP);
      fas.allocate_field(sizeof(double), FID_SVOLP);
      fas.allocate_field(sizeof(double2), FID_SSURFP);
      fas.allocate_field(sizeof(double), FID_ELEN);
      fas.allocate_field(sizeof(double), FID_SMF);
      fas.allocate_field(sizeof(double2), FID_SFP);
      fas.allocate_field(sizeof(double2), FID_SFQ);
      fas.allocate_field(sizeof(double2), FID_SFT);
      fas.allocate_field(sizeof(double), FID_CAREA);
      fas.allocate_field(sizeof(double), FID_CEVOL);
      fas.allocate_field(sizeof(double), FID_CDU);
      fas.allocate_field(sizeof(double), FID_CDIV);
      fas.allocate_field(sizeof(double), FID_CCOS);
      fas.allocate_field(sizeof(double2), FID_CQE1);
      fas.allocate_field(sizeof(double2), FID_CQE2);
      fas.allocate_field(sizeof(double), FID_CRMU);
      fas.allocate_field(sizeof(double), FID_CW);
    }
    lrs = runtime->create_logical_region(ctx, iss, fss);
    runtime->attach_name(lrs, "lrs");
    IndexPartition equal_sides = runtime->create_equal_partition(ctx, iss, piece_is);
    lps = runtime->get_logical_partition(lrs, equal_sides);
    // construct temp side maps with equal partition (iterate over zones and find sides)
    gmesh->generateSideMapsParallel(numpcs, runtime, ctx, lrs, lps);
    // Now we need to compact our points and generate our point partition tree
    // First compute our owned points
    IndexPartition owned_points = runtime->create_partition_by_field(ctx, lrp, lrp,
                                                              FID_PIECE, piece_is);
    runtime->attach_name(owned_points, "owned points");
    // Now find the set of points that we can reach from our points through all our sides
    IndexPartition owned_sides = runtime->create_partition_by_preimage(ctx, owned_points,
                                                         lrs, lrs, FID_MAPSP1, piece_is);
    IndexPartition reachable_points = runtime->create_partition_by_image(ctx, isp,
            runtime->get_logical_partition(lrs, owned_sides), lrs, FID_MAPSP2, piece_is);
    runtime->attach_name(reachable_points, "reachable points");
    // Now we can make the temp ghost partition
    IndexPartition temp_ghost_points = runtime->create_partition_by_difference(ctx,
                                          isp, reachable_points, owned_points, piece_is);
    runtime->attach_name(temp_ghost_points, "temporary ghost points");
    // Now create a two-way partition of private versus shared
    IndexSpace private_is = runtime->create_index_space(ctx, Rect<1>(0, 1));
    IndexPartition private_all = runtime->create_pending_partition(ctx, iss, private_is);
    // Fill in the two sub-regions of the ipp index space
    IndexSpace all_shared = runtime->create_index_space_union(ctx, private_all,
                                      Point<1>(1)/*color*/, temp_ghost_points);
    std::vector<IndexSpace> diff_spaces(1, all_shared);
    IndexSpace all_private = runtime->create_index_space_difference(ctx, private_all, 
                                      Point<1>(0)/*color*/, isp, diff_spaces);
    runtime->attach_name(private_all, "temporary private");
    // create the private and shared partitions with cross product partitions
    // There are only going to be two of them so we can get their names back
    // right away without having to worry about scalability
    std::map<IndexSpace,IndexPartition> partition_handles;
    partition_handles[all_shared] = IndexPartition::NO_PART;
    partition_handles[all_private] = IndexPartition::NO_PART;
    runtime->create_cross_product_partitions(ctx, private_all, 
                                owned_points, partition_handles);
    IndexPartition temp_shared = partition_handles[all_shared];
    IndexPartition temp_private = partition_handles[all_private];
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
    LogicalRegion lr_all_range = runtime->create_logical_region(ctx, private_is, fsc);
    LogicalRegion lr_private_range = runtime->create_logical_region(ctx, piece_is, fsc);
    LogicalPartition lp_private_range = 
      runtime->get_logical_partition(lr_private_range, piece_ip);
    LogicalRegion lr_shared_range = runtime->create_logical_region(ctx, piece_is, fsc);
    LogicalPartition lp_shared_range = 
      runtime->get_logical_partition(lr_shared_range, piece_ip);
    computeRangesParallel(numpcs, runtime, ctx, lr_all_range,
        lr_private_range, lp_private_range, lr_shared_range, lp_shared_range,
        temp_private, temp_shared);
    // Now we can compute the actual dense versions of the partition
    IndexPartition private_ip = runtime->create_equal_partition(ctx, private_is, private_is);
    IndexPartition ippall = runtime->create_partition_by_image_range(ctx, isp,
        runtime->get_logical_partition(lr_all_range, private_ip), lr_all_range, 
        FID_RANGE, private_is);
    IndexSpace ispprv = runtime->get_index_subspace(ippall, DomainPoint(0));
    IndexSpace ispshr = runtime->get_index_subspace(ippall, DomainPoint(1));
    IndexPartition ippprv = runtime->create_partition_by_image_range(ctx, ispprv,
        lp_private_range, lr_private_range, FID_RANGE, piece_is);
    IndexPartition ippmstr = runtime->create_partition_by_image_range(ctx, ispshr,
        lp_shared_range, lr_shared_range, FID_RANGE, piece_is);
    // Now compute the actual ghost region by projecting through the sides
    IndexPartition shared_sides = runtime->create_partition_by_preimage(ctx, ippmstr,
                                                    lrs, lrs, FID_MAPSP1, piece_is);
    IndexPartition reachable_ghost_points = runtime->create_partition_by_image(ctx, ispshr,
            runtime->get_logical_partition(lrs, shared_sides), lrs, FID_MAPSP2, piece_is);
    IndexPartition ippshr = runtime->create_partition_by_difference(ctx, ispshr,
                                      reachable_ghost_points, ippmstr, piece_is);
    // Now make the actual point logical region, get the partitions, and copy over data
    LogicalRegion lrp = runtime->create_logical_region(ctx, isp, fsp);
    runtime->attach_name(lrp, "lrp");
    lppprv = runtime->get_logical_partition_by_tree(ippprv, fsp, lrp.get_tree_id());
    runtime->attach_name(lppprv, "lppprv");
    lppmstr = runtime->get_logical_partition_by_tree(ippmstr, fsp, lrp.get_tree_id());
    runtime->attach_name(lppmstr, "lppmstr");
    lppshr = runtime->get_logical_partition_by_tree(ippshr, fsp, lrp.get_tree_id());
    runtime->attach_name(lppshr, "lppshr");
    // Compact the points
    compactPointsParallel(numpcs, runtime, ctx, temp_points, 
        runtime->get_logical_partition_by_tree(ippprv, fsp, temp_points.get_tree_id()),
        lrp, lppprv);
    compactPointsParallel(numpcs, runtime, ctx, temp_points,
        runtime->get_logical_partition_by_tree(ippmstr, fsp, temp_points.get_tree_id()),
        lrp, lppmstr);

    // Calculate centers, volumes, and side fractions

    // Delete our temporary regions
    runtime->destroy_logical_region(ctx, temp_points);
    runtime->destroy_logical_region(ctx, lr_all_range);
    runtime->destroy_logical_region(ctx, lr_private_range);
    runtime->destroy_logical_region(ctx, lr_shared_range);
}


void Mesh::initSides(
        std::vector<int>& cellstart,
        std::vector<int>& cellsize,
        std::vector<int>& cellnodes) {

    mapsp1 = alloc<int>(nums);
    mapsp2 = alloc<int>(nums);
    mapsz  = alloc<int>(nums);
    mapss3 = alloc<int>(nums);
    mapss4 = alloc<int>(nums);

    for (int z = 0; z < numz; ++z) {
        int sbase = cellstart[z];
        int size = cellsize[z];
        for (int n = 0; n < size; ++n) {
            int s = sbase + n;
            int snext = sbase + (n + 1 == size ? 0 : n + 1);
            int slast = sbase + (n == 0 ? size : n) - 1;
            mapsz[s] = z;
            mapsp1[s] = cellnodes[s];
            mapsp2[s] = cellnodes[snext];
            mapss3[s] = slast;
            mapss4[s] = snext;
        } // for n
    } // for z

}


void Mesh::initChunks() {

    if (chunksize == 0) chunksize = max(nump, nums);
    // check for bad chunksize
    if (chunksize < 0) {
        cerr << "Error: bad chunksize " << chunksize << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

    // compute side chunks
    // use 'chunksize' for maximum chunksize; decrease as needed
    // to ensure that no zone has its sides split across chunk
    // boundaries
    int s1, s2 = 0;
    while (s2 < nums) {
        s1 = s2;
        s2 = min(s2 + chunksize, nums);
        while (s2 < nums && mapsz[s2] == mapsz[s2-1])
            --s2;
        schsfirst.push_back(s1);
        schslast.push_back(s2);
        schzfirst.push_back(mapsz[s1]);
        schzlast.push_back(mapsz[s2-1] + 1);
    }
    numsch = schsfirst.size();

    // compute point chunks
    int p1, p2 = 0;
    while (p2 < nump) {
        p1 = p2;
        p2 = min(p2 + chunksize, nump);
        pchpfirst.push_back(p1);
        pchplast.push_back(p2);
    }
    numpch = pchpfirst.size();

    // compute zone chunks
    int z1, z2 = 0;
    while (z2 < numz) {
        z1 = z2;
        z2 = min(z2 + chunksize, numz);
        zchzfirst.push_back(z1);
        zchzlast.push_back(z2);
    }
    numzch = zchzfirst.size();

}


void Mesh::writeStats() {

    int gnump = nump;
    int gnumz = numz;
    int gnums = nums;
    int gnumpch = numpch;
    int gnumzch = numzch;
    int gnumsch = numsch;

    cout << "--- Mesh Information ---" << endl;
    cout << "Points:  " << gnump << endl;
    cout << "Zones:  "  << gnumz << endl;
    cout << "Sides:  "  << gnums << endl;
    cout << "Side chunks:  " << gnumsch << endl;
    cout << "Point chunks:  " << gnumpch << endl;
    cout << "Zone chunks:  " << gnumzch << endl;
    cout << "Chunk size:  " << chunksize << endl;
    cout << "------------------------" << endl;

}


void Mesh::write(
        const string& probname,
        const int cycle,
        const double time,
        const double* zr,
        const double* ze,
        const double* zp) {

    wxy->write(probname, zr, ze, zp);
    egold->write(probname, cycle, time, zr, ze, zp);

}


vector<int> Mesh::getXPlane(const double c) {

    vector<int> mapbp;
    const double eps = 1.e-12;

    for (int p = 0; p < nump; ++p) {
        if (fabs(px[p].x - c) < eps) {
            mapbp.push_back(p);
        }
    }
    return mapbp;

}


vector<int> Mesh::getYPlane(const double c) {

    vector<int> mapbp;
    const double eps = 1.e-12;

    for (int p = 0; p < nump; ++p) {
        if (fabs(px[p].y - c) < eps) {
            mapbp.push_back(p);
        }
    }
    return mapbp;

}


void Mesh::getPlaneChunks(
        const int numb,
        const int* mapbp,
        vector<int>& pchbfirst,
        vector<int>& pchblast) {

    pchbfirst.resize(0);
    pchblast.resize(0);

    // compute boundary point chunks
    // (boundary points contained in each point chunk)
    int bf, bl = 0;
    for (int pch = 0; pch < numpch; ++pch) {
         int pl = pchplast[pch];
         bf = bl;
         bl = lower_bound(&mapbp[bf], &mapbp[numb], pl) - &mapbp[0];
         pchbfirst.push_back(bf);
         pchblast.push_back(bl);
    }

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


void Mesh::calcVols(
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

    numsbad += count;

}


void Mesh::checkBadSides() {

    // if there were negative side volumes, error exit
    numsbad = reduceFutureMap<SumOp<int> >(fmapcv);
    if (numsbad > 0) {
        cerr << "Error: " << numsbad << " negative side volumes" << endl;
        cerr << "Exiting..." << endl;
        exit(1);
    }

}


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



