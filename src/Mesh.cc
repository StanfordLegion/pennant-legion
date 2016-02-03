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
#include "legion_io.h"


#include "MyLegion.hh"
#include "Vec2.hh"
#include "Memory.hh"
#include "InputFile.hh"
#include "GenMesh.hh"
#include "WriteXY.hh"
#include "ExportGold.hh"

using namespace std;
using namespace Memory;
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


namespace {  // unnamed
static void __attribute__ ((constructor)) registerTasks() {
    HighLevelRuntime::register_legion_task<Mesh::copyFieldTask<double> >(
            TID_COPYFIELDDBL, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "copyfielddbl");
    HighLevelRuntime::register_legion_task<Mesh::copyFieldTask<double2> >(
            TID_COPYFIELDDBL2, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "copyfielddbl2");
    HighLevelRuntime::register_legion_task<Mesh::fillFieldTask<double> >(
            TID_FILLFIELDDBL, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "fillfielddbl");
    HighLevelRuntime::register_legion_task<Mesh::fillFieldTask<double2> >(
            TID_FILLFIELDDBL2, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "fillfielddbl2");
    HighLevelRuntime::register_legion_task<Mesh::calcCtrsTask>(
            TID_CALCCTRS, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "calcctrs");
    HighLevelRuntime::register_legion_task<int, Mesh::calcVolsTask>(
            TID_CALCVOLS, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "calcvols");
    HighLevelRuntime::register_legion_task<Mesh::calcSurfVecsTask>(
            TID_CALCSURFVECS, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "calcsurfvecs");
    HighLevelRuntime::register_legion_task<Mesh::calcEdgeLenTask>(
            TID_CALCEDGELEN, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "calcedgelen");
    HighLevelRuntime::register_legion_task<Mesh::calcCharLenTask>(
            TID_CALCCHARLEN, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "calccharlen");

    HighLevelRuntime::register_reduction_op<SumOp<int> >(
            OPID_SUMINT);
    HighLevelRuntime::register_reduction_op<SumOp<double> >(
            OPID_SUMDBL);
    HighLevelRuntime::register_reduction_op<SumOp<double2> >(
            OPID_SUMDBL2);
    HighLevelRuntime::register_reduction_op<MinOp<double> >(
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
        Context ctxa,
        HighLevelRuntime* runtimea)
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
    IndexAllocator iap = runtime->create_index_allocator(ctx, isp);
    iap.alloc(nump);
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
    num_h = 1; // GMS HACK
    int num_elements = nump * num_h; 
    IndexSpace p_isp = runtime->create_index_space(ctx, num_elements);
    IndexAllocator p_iap = runtime->create_index_allocator(ctx, p_isp);
    p_iap.alloc(num_elements); 
    p_lrp = runtime->create_logical_region(ctx, p_isp, fsp);
    runtime->attach_name(p_lrp, "persistent points LR");
    int num_subregions = 4; // GMS HACK 
    IndexPartition ipp;
    Coloring hdf_parts;
    int num_per_part = num_elements / num_subregions;
    IndexIterator itr(runtime, ctx, p_isp); 
    const int lower_bound = num_elements / num_subregions;
    const int upper_bound = lower_bound+1;
    const int number_small = num_subregions - (num_elements % num_subregions); 
    for(int color = 0; color < num_subregions; color++) {
      int cur_num_elements = color < number_small ? lower_bound : upper_bound; 
      for(int elem = 0; elem < cur_num_elements; elem++) { 
        assert(itr.has_next());
        ptr_t point_ptr = itr.next();
        hdf_parts[color].points.insert(point_ptr);
      }
    }
    ipp = runtime->create_index_partition(ctx, p_isp, hdf_parts, true); 
    
    // if ((num_elements % num_subregions) != 0) {
    //   // Not evenly divisible
    //   const int lower_bound = num_elements/num_subregions;
    //   const int upper_bound = lower_bound+1;
    //   const int number_small = num_subregions - (num_elements % num_subregions);
    //   DomainColoring coloring;
    //   int index = 0;
    //   for (int color = 0; color < num_subregions; color++) {
        
    //     int num_elmts = color < number_small ? lower_bound : upper_bound;
    //     assert((index+num_elmts) <= num_elements);
    //     Rect<1> subrect(Point<1>(index),Point<1>(index+num_elmts-1));
    //     coloring[color] = Domain::from_rect<1>(subrect);
    //     index += num_elmts;
    //   }
    //   ipp = runtime->create_index_partition(ctx, p_isp, color_domain,
    //                                         coloring, true/*disjoint*/);
    // } else {
    //   Blockify<1> coloring(num_elements/num_subregions);
    //   ipp = runtime->create_index_partition(ctx, p_isp, coloring);
    // }
    
    
    LogicalPartition p_lpp =
      runtime->get_logical_partition(ctx, p_lrp, ipp); 
                                                                    
                                                                  
    PersistentRegion points_pr = PersistentRegion(runtime);
    std::map<FieldID, std::string> field_string_map;
    field_string_map.insert(std::make_pair(FID_PX, "bam/px"));
    field_string_map.insert(std::make_pair(FID_PXP, "bam/pxp"));
    field_string_map.insert(std::make_pair(FID_PX0, "bam/px0"));
    field_string_map.insert(std::make_pair(FID_PU, "bam/pu"));
    field_string_map.insert(std::make_pair(FID_PU0, "bam/pu0"));
    field_string_map.insert(std::make_pair(FID_PMASWT, "bam/pmaswt"));
    field_string_map.insert(std::make_pair(FID_PF, "bam/pf"));
    field_string_map.insert(std::make_pair(FID_PAP, "bam/pap"));
    
    
    points_pr.create_persistent_subregions(ctx, "points_pr.hdf5",
                                           p_lrp, p_lpp, hdf_parts,
                                           field_string_map);
    
    
    
    IndexSpace isz = runtime->create_index_space(ctx, numz);
    IndexAllocator iaz = runtime->create_index_allocator(ctx, isz);
    iaz.alloc(numz);
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
    IndexAllocator ias = runtime->create_index_allocator(ctx, iss);
    ias.alloc(nums);
    FieldSpace fss = runtime->create_field_space(ctx);
    FieldAllocator fas = runtime->create_field_allocator(ctx, fss);
    fas.allocate_field(sizeof(ptr_t), FID_MAPSP1);
    fas.allocate_field(sizeof(ptr_t), FID_MAPSP2);
    fas.allocate_field(sizeof(ptr_t), FID_MAPSZ);
    fas.allocate_field(sizeof(ptr_t), FID_MAPSS3);
    fas.allocate_field(sizeof(ptr_t), FID_MAPSS4);
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
    IndexAllocator iaglb = runtime->create_index_allocator(ctx, isglb);
    iaglb.alloc(1);
    FieldSpace fsglb = runtime->create_field_space(ctx);
    FieldAllocator faglb = runtime->create_field_allocator(ctx, fsglb);
    faglb.allocate_field(sizeof(int), FID_NUMSBAD);
    faglb.allocate_field(sizeof(double), FID_DTREC);
    lrglb = runtime->create_logical_region(ctx, isglb, fsglb);
    runtime->attach_name(lrglb, "lrglb");

    // create domain over pieces
    Rect<1> task_rect(Point<1>(0), Point<1>(numpcs-1));
    dompc  = Domain::from_rect<1>(task_rect);
    
    
#if 0 
    IndexSpace ispc = runtime->create_index_space(ctx, dompc);
    {
      
      IndexAllocator allocator = runtime->create_index_allocator(ctx, ispc);
      allocator.alloc(numpcs);
    }
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

    vector<ptr_t> lgmapsp1(&mapsp1[0], &mapsp1[nums]);
    vector<ptr_t> lgmapsp2(&mapsp2[0], &mapsp2[nums]);
    vector<ptr_t> lgmapsz (&mapsz [0], &mapsz [nums]);
    vector<ptr_t> lgmapss3(&mapss3[0], &mapss3[nums]);
    vector<ptr_t> lgmapss4(&mapss4[0], &mapss4[nums]);

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


template <typename T>
void Mesh::copyFieldTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
  // determine which fields to use in the copy
    FieldID fid_src = *(task->regions[0].instance_fields.begin());
    FieldID fid_dst = *(task->regions[1].instance_fields.begin());
    LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::SOA<sizeof(T)> , T> acc_src;
    acc_src = regions[0].get_field_accessor(fid_src).typeify<T>(). template convert<LegionRuntime::Accessor::AccessorType::SOA<sizeof(T)> >(); 
      
    //MyAccessor<T> acc_src = get_accessor<T>(regions[0], fid_src); //regions[0].get_field_accessor(fid_src).typeify<T>().convert<AccessorType::SOA>();
    LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::SOA<sizeof(T)> , T> acc_dst;
    acc_dst = regions[1].get_field_accessor(fid_dst).typeify<T>(). template convert<LegionRuntime::Accessor::AccessorType::SOA<sizeof(T)> >(); 
//    MyAccessor<T> acc_dst =
//        get_accessor<T>(regions[1], fid_dst);

    const IndexSpace& is = task->regions[0].region.get_index_space();
    for (IndexIterator itr(runtime, ctx, is); itr.has_next(); )
    {
        ptr_t idx = itr.next();
        acc_dst.write(idx, acc_src.read(idx));
    }

}


template <typename T>
void Mesh::fillFieldTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
    const T* args = (const T*) task->args;
    const T val = args[0];

    FieldID fid_var = task->regions[0].instance_fields[0];
   LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::SOA<sizeof(T)> , T> acc_var;
   acc_var = regions[0].get_field_accessor(fid_var).typeify<T>(). template convert<LegionRuntime::Accessor::AccessorType::SOA<sizeof(T)> >(); 
      
//     MyAccessor<T> acc_var =
//        get_accessor<T>(regions[0], fid_var);

    const IndexSpace& is = task->regions[0].region.get_index_space();
    
    for (IndexIterator itr(runtime, ctx, is); itr.has_next();)
    {
        ptr_t idx = itr.next();
        acc_var.write(idx, val);
    }

}


void Mesh::calcCtrsTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
    MyAccessor<ptr_t> acc_mapsp1 =
        get_accessor<ptr_t>(regions[0], FID_MAPSP1);
    MyAccessor<ptr_t> acc_mapsp2 =
        get_accessor<ptr_t>(regions[0], FID_MAPSP2);
    MyAccessor<ptr_t> acc_mapsz =
        get_accessor<ptr_t>(regions[0], FID_MAPSZ);
    MyAccessor<int> acc_mapsp1reg =
        get_accessor<int>(regions[0], FID_MAPSP1REG);
    MyAccessor<int> acc_mapsp2reg =
        get_accessor<int>(regions[0], FID_MAPSP2REG);
    MyAccessor<int> acc_znump =
        get_accessor<int>(regions[1], FID_ZNUMP);
    FieldID fid_px = task->regions[2].instance_fields[0];
    MyAccessor<double2> acc_px[2] = {
        get_accessor<double2>(regions[2], fid_px),
        get_accessor<double2>(regions[3], fid_px)
    };
    FieldID fid_ex = task->regions[4].instance_fields[0];
    MyAccessor<double2> acc_ex =
        get_accessor<double2>(regions[4], fid_ex);
    FieldID fid_zx = task->regions[5].instance_fields[0];
    MyAccessor<double2> acc_zx =
        get_accessor<double2>(regions[5], fid_zx);

    const IndexSpace& isz = task->regions[1].region.get_index_space();
    for (IndexIterator itrz(runtime,ctx,isz); itrz.has_next();)
    {
        ptr_t z = itrz.next();
        acc_zx.write(z, double2(0., 0.));

    }

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (IndexIterator itrs(runtime,ctx, iss); itrs.has_next(); )
    {
        ptr_t s = itrs.next();
        ptr_t p1 = acc_mapsp1.read(s);
        int p1reg = acc_mapsp1reg.read(s);
        ptr_t p2 = acc_mapsp2.read(s);
        int p2reg = acc_mapsp2reg.read(s);
        ptr_t z  = acc_mapsz.read(s);
        double2 px1 = acc_px[p1reg].read(p1);
        double2 px2 = acc_px[p2reg].read(p2);
        double2 ex  = 0.5 * (px1 + px2);
        acc_ex.write(s, ex);
        double2 zx  = acc_zx.read(z);
        int n = acc_znump.read(z);
        zx += px1 / n;
        acc_zx.write(z, zx);
    }
}


int Mesh::calcVolsTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
    MyAccessor<ptr_t> acc_mapsp1 =
        get_accessor<ptr_t>(regions[0], FID_MAPSP1);
    MyAccessor<ptr_t> acc_mapsp2 =
        get_accessor<ptr_t>(regions[0], FID_MAPSP2);
    MyAccessor<ptr_t> acc_mapsz =
        get_accessor<ptr_t>(regions[0], FID_MAPSZ);
    MyAccessor<int> acc_mapsp1reg =
        get_accessor<int>(regions[0], FID_MAPSP1REG);
    MyAccessor<int> acc_mapsp2reg =
        get_accessor<int>(regions[0], FID_MAPSP2REG);
    FieldID fid_px = task->regions[1].instance_fields[0];
    MyAccessor<double2> acc_px[2] = {
        get_accessor<double2>(regions[1], fid_px),
        get_accessor<double2>(regions[2], fid_px)
    };
    FieldID fid_zx = task->regions[3].instance_fields[0];
    MyAccessor<double2> acc_zx =
        get_accessor<double2>(regions[3], fid_zx);
    FieldID fid_sarea = task->regions[4].instance_fields[0];
    FieldID fid_svol  = task->regions[4].instance_fields[1];
    MyAccessor<double> acc_sarea =
        get_accessor<double>(regions[4], fid_sarea);
    MyAccessor<double> acc_svol =
        get_accessor<double>(regions[4], fid_svol);
    FieldID fid_zarea = task->regions[5].instance_fields[0];
    FieldID fid_zvol  = task->regions[5].instance_fields[1];
    MyAccessor<double> acc_zarea =
        get_accessor<double>(regions[5], fid_zarea);
    MyAccessor<double> acc_zvol =
        get_accessor<double>(regions[5], fid_zvol);

    const IndexSpace& isz = task->regions[3].region.get_index_space();
    for (IndexIterator itrz(runtime, ctx, isz); itrz.has_next(); )
    {
        ptr_t z = itrz.next();
        acc_zarea.write(z, 0.);
        acc_zvol.write(z, 0.);
    }

    const double third = 1. / 3.;
    int count = 0;
    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (IndexIterator itrs(runtime, ctx, iss); itrs.has_next(); )
    {
        ptr_t s = itrs.next();
        ptr_t p1 = acc_mapsp1.read(s);
        int p1reg = acc_mapsp1reg.read(s);
        ptr_t p2 = acc_mapsp2.read(s);
        int p2reg = acc_mapsp2reg.read(s);
        ptr_t z  = acc_mapsz.read(s);
        double2 px1 = acc_px[p1reg].read(p1);
        double2 px2 = acc_px[p2reg].read(p2);
        double2 zx  = acc_zx.read(z);

        // compute side volumes, sum to zone
        double sa = 0.5 * cross(px2 - px1, zx - px1);
        double sv = third * sa * (px1.x + px2.x + zx.x);
        acc_sarea.write(s, sa);
        acc_svol.write(s, sv);
        double za = acc_zarea.read(z);
        za += sa;
        acc_zarea.write(z, za);
        double zv = acc_zvol.read(z);
        zv += sv;
        acc_zvol.write(z, zv);

        // check for negative side volumes
        if (sv <= 0.) count += 1;
    }

    return count;
}


void Mesh::calcSurfVecsTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
    MyAccessor<ptr_t> acc_mapsz =
        get_accessor<ptr_t>(regions[0], FID_MAPSZ);
    MyAccessor<double2> acc_ex =
        get_accessor<double2>(regions[0], FID_EXP);
    MyAccessor<double2> acc_zx =
        get_accessor<double2>(regions[1], FID_ZXP);
    MyAccessor<double2> acc_ssurf =
        get_accessor<double2>(regions[2], FID_SSURFP);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (IndexIterator itrs(runtime,ctx,iss); itrs.has_next();)
    {
        ptr_t s = itrs.next();
        ptr_t z = acc_mapsz.read(s);
        double2 ex = acc_ex.read(s);
        double2 zx = acc_zx.read(z);
        double2 ss = rotateCCW(ex - zx);
        acc_ssurf.write(s, ss);
    }
}


void Mesh::calcEdgeLenTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
    MyAccessor<ptr_t> acc_mapsp1 =
        get_accessor<ptr_t>(regions[0], FID_MAPSP1);
    MyAccessor<ptr_t> acc_mapsp2 =
        get_accessor<ptr_t>(regions[0], FID_MAPSP2);
    MyAccessor<int> acc_mapsp1reg =
        get_accessor<int>(regions[0], FID_MAPSP1REG);
    MyAccessor<int> acc_mapsp2reg =
        get_accessor<int>(regions[0], FID_MAPSP2REG);
    MyAccessor<double2> acc_px[2] = {
        get_accessor<double2>(regions[1], FID_PXP),
        get_accessor<double2>(regions[2], FID_PXP)
    };
    MyAccessor<double> acc_elen =
        get_accessor<double>(regions[3], FID_ELEN);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (IndexIterator itrs(runtime,ctx,iss); itrs.has_next(); )
    {
        ptr_t s = itrs.next();
        ptr_t p1 = acc_mapsp1.read(s);
        int p1reg = acc_mapsp1reg.read(s);
        ptr_t p2 = acc_mapsp2.read(s);
        int p2reg = acc_mapsp2reg.read(s);
        double2 px1 = acc_px[p1reg].read(p1);
        double2 px2 = acc_px[p2reg].read(p2);

        double elen = length(px2 - px1);
        acc_elen.write(s, elen);
    }
}


void Mesh::calcCharLenTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
    MyAccessor<ptr_t> acc_mapsz =
        get_accessor<ptr_t>(regions[0], FID_MAPSZ);
    MyAccessor<double> acc_elen =
        get_accessor<double>(regions[0], FID_ELEN);
    MyAccessor<double> acc_sarea =
        get_accessor<double>(regions[0], FID_SAREAP);
    MyAccessor<int> acc_znump =
        get_accessor<int>(regions[1], FID_ZNUMP);
    MyAccessor<double> acc_zdl =
        get_accessor<double>(regions[2], FID_ZDL);

    const IndexSpace& isz = task->regions[1].region.get_index_space();
    for (IndexIterator itrz(runtime,ctx,isz); itrz.has_next();)
    {
        ptr_t z = itrz.next();
        acc_zdl.write(z, 1.e99);
        
    }
    
    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (IndexIterator itrs(runtime,ctx,iss); itrs.has_next();)
    {
        ptr_t s = itrs.next();
        ptr_t z  = acc_mapsz.read(s);
        double area = acc_sarea.read(s);
        double base = acc_elen.read(s);
        double zdl = acc_zdl.read(z);
        int np = acc_znump.read(z);
        double fac = (np == 3 ? 3. : 4.);
        double sdl = fac * area / base;
        zdl = min(zdl, sdl);
        acc_zdl.write(z, zdl);
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

//    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int z = mapsz[s];
        smf[s] = sarea[s] / zarea[z];
    }
}



