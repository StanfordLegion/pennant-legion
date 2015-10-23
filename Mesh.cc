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
#include "Utils.hh"

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
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


namespace {  // unnamed
static void __attribute__ ((constructor)) registerTasks() {
  HighLevelRuntime::register_legion_task<Mesh::SPMDtask>(
    TID_STENCIL, Processor::LOC_PROC, true, true,
    AUTO_GENERATE_ID, TaskConfigOptions(true), "stencilTask");
  
  HighLevelRuntime::register_legion_task<Mesh::SPMDtask>(
    TID_SPMD_TASK, Processor::LOC_PROC, true, true,
    AUTO_GENERATE_ID, TaskConfigOptions(true), "SPMDtask"); 
  
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
    vector<int> zonestart, zonesize, zonenodes;
    vector<int> zonecolors;
    gmesh->generate(numpcs, nodepos, pointcolors, pointmcolors,
            zonestart, zonesize, zonenodes, zonecolors);

    nump = nodepos.size();
    numz = zonestart.size();
    nums = zonenodes.size();
    numc = nums;

    // copy zone sizes to mesh
    znump = alloc<int>(numz);
    copy(zonesize.begin(), zonesize.end(), znump);

    // populate maps:
    // use the zone* arrays to populate the side maps
    initSides(zonestart, zonesize, zonenodes);
    // release memory from zone* arrays
    zonestart.resize(0);
    zonesize.resize(0);
    zonenodes.resize(0);

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

    /* setup explicit ghost fields rather than simply sharing fields */ 
    FieldSpace fsp_ghost = runtime->create_field_space(ctx);
    FieldAllocator fap_ghost = runtime->create_field_allocator(ctx, fsp_ghost);
    fap_ghost.allocate_field(sizeof(double2), FID_PXP_GHOST);
    fap_ghost.allocate_field(sizeof(double2), FID_PU_GHOST);
    fap_ghost.allocate_field(sizeof(double2), FID_PU0_GHOST);
    fap_ghost.allocate_field(sizeof(double), FID_PMASWT_GHOST);
    fap_ghost.allocate_field(sizeof(double2), FID_PF_GHOST);
    
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
    IndexSpace ispc = runtime->create_index_space(ctx, dompc);
    runtime->attach_name(ispc, "disjoint index space");


    // create zone and side partitions
    Coloring colorz, colors;
    int z0 = 0;
    while (z0 < numz)
    {
        int s0 = zonestart[z0];
        int c = zonecolors[z0];
        int z1 = z0 + 1;
        // find range with [z0, z1) all the same color
        while (z1 < numz && zonecolors[z1] == c) ++z1;
        int s1 = (z1 < numz ? zonestart[z1] : nums);
        colorz[c].ranges.insert(pair<int, int>(z0, z1 - 1));
        colors[c].ranges.insert(pair<int, int>(s0, s1 - 1));
        DEBUG( "Adding range of zones  " << z0 << "," << z1-1 << " to color: " << c << std::endl);
        DEBUG( "Adding range of sides  " << s0 << "," << s1-1 << " to color: " << c << std::endl);
               
        z0 = z1;
    }
    IndexPartition ipz = runtime->create_index_partition(
                ctx, isz, colorz, true);
    lpz = runtime->get_logical_partition(ctx, lrz, ipz);
    IndexPartition ips = runtime->create_index_partition(
                ctx, iss, colors, true);
    lps = runtime->get_logical_partition(ctx, lrs, ips);

    // create point partitions
    Coloring colorpall, colorpprv, colorpshr, colorpmstr, colorpmasterghost, colorpslaveghost;  
    // force all colors to exist, even if they might be empty
    colorpall[0];
    colorpall[1];
    for (int c = 0; c < numpcs; ++c) {
        colorpprv[c];
        colorpmstr[c];
        colorpshr[c];
#ifdef PENNANT_DEPENDENT_PARTITIONING
        colorpghost[c];
#endif
    }
    /* create colorpace with all possible two color combos */ 
    for (int c = 0; c < numpcs*numpcs; ++c) {
      colorpslaveghost[c];
      colorpmasterghost[c];  
    }
    /* now create a color space that will be used to identify each intersection of
       colors, for ghosting */
    Domain color_space_ghost = Domain::from_rect<1>(Rect<1>(0, (numpcs * numpcs)-1)); 

    
    int p0 = 0;
    while (p0 < nump)
    {
        int c = pointcolors[p0];
        int p1 = p0 + 1;
        if (c != MULTICOLOR) {
            // find range with [p0, p1) all the same color
            while (p1 < nump && pointcolors[p1] == c) ++p1;
            colorpall[0].ranges.insert(pair<int, int>(p0, p1 - 1));
            colorpprv[c].ranges.insert(pair<int, int>(p0, p1 - 1));
        }
        else {
            // insert p0 by itself
            colorpall[1].points.insert(p0);
            vector<int>& pmc = pointmcolors[p0];
            colorpmstr[pmc[0]].points.insert(p0);
            for (int i = 0; i < pmc.size(); ++i) {
              colorpshr[pmc[i]].points.insert(p0);
              if(i > 0) {
                /* only add true ghosts to the ghost coloring
                   note that we use the convention that the color of the ghost regions 
                   is the master color + the slave color with the slave color scaled by the 
                   the width of the color space 
                */
                DEBUG( "Adding a point to master ghost color " << pmc[0] << "," << pmc[i] << std::endl ); 
                colorpmasterghost[pmc[0]+pmc[i]*numpcs].points.insert(p0);
                DEBUG( "Adding a point to slave ghost color " << pmc[i] << "," << pmc[0] << std::endl ); 
                colorpslaveghost[pmc[i]+pmc[0]*numpcs].points.insert(p0);
              }
            }
            // Create logical partitions for ghosts
            
        }
        p0 = p1;
    }


    int nshared_points = 0;
    char buf[32]; 
    for (int c = 0; c < numpcs; ++c){
      DEBUG("Color " << c << " has " << colorpshr[c].points.size() << " shared points" << std::endl);
      /* create a ghost region for each of these colors */
      nshared_points += colorpshr[c].points.size(); 
    }
    DEBUG (" Total number of shared points is: " << pointmcolors.size() << " and color*point total is " << nshared_points << std::endl ); 

    typedef std::set<int> my_set;
    typedef std::map<int, my_set> my_map_set;
    my_map_set map_master_to_slave;
    my_map_set map_slaves_to_masters;
    /* this is a hack ghould be moved to GenMesh::generate.. */
    for(int pt = 0; pt < pointmcolors.size(); pt++) {
      //DEBUG("PT: " << pt << std::endl);
      vector<int> &pmc = pointmcolors[pt];
      if(!pmc.size()) continue;
      my_map_set::iterator it_master = map_master_to_slave.find(pmc[0]);
      if(it_master==map_master_to_slave.end()){
        my_set slaves;
        map_master_to_slave.insert(std::pair<int, my_set>(pmc[0], slaves));
        it_master = map_master_to_slave.find(pmc[0]);
      }
      DEBUG("On pt: " << pt << " master color is " << pmc[0] << " additional colors are: "); 
      for (int i = 1; i < pmc.size(); i++) {
        DEBUG(pmc[i] << ", ");
        it_master->second.insert(pmc[i]);
        my_map_set::iterator it = map_slaves_to_masters.find(pmc[i]);
        if(it!=map_slaves_to_masters.end()) {
          it->second.insert(pmc[0]);
        } else {
          my_set masters;
          masters.insert(pmc[0]); 
          map_slaves_to_masters.insert(std::pair<int, my_set>(pmc[i], masters));
        }
      }
      DEBUG(std::endl);
    }

    for(my_map_set::iterator it = map_master_to_slave.begin(); it != map_master_to_slave.end(); it++) {
      DEBUG("Master is: " << it->first << " slave(s) are: ");
      // Create two phase barriers for each intersection of colors, one for
      //  master to tell slaves when the new point values are available and
      //  another for slaves to tell masters when it is safe to update points in ghost 
      for(my_set::iterator its = it->second.begin(); its != it->second.end(); its++){
        ready_barriers[it->first][*its] = runtime->create_phase_barrier(ctx, 1);
        empty_barriers[it->first][*its] = runtime->create_phase_barrier(ctx, 1);
        DEBUG(*its << ","); 
      }
      DEBUG(std::endl);
    }
    
    for(my_map_set::iterator it = map_slaves_to_masters.begin(); it != map_slaves_to_masters.end(); it++) {
      DEBUG("Slave is: " << it->first << " master(s) are: ");
      for(my_set::iterator its = it->second.begin(); its != it->second.end(); its++){
        DEBUG(*its << ","); 
      }
      DEBUG(std::endl); 
    }

    for(int pt = 0; pt < pointmcolors.size(); pt++) {
      vector<int> &pmc = pointmcolors[pt];
      if(!pmc.size()) continue; 
      DEBUG("Point " << pt << " has colors: ");
      for (int i = 0; i < pmc.size(); i++) {
        DEBUG(pmc[i] << ",");
        
      }
      DEBUG(std::endl);
        
    }
    
    /* this is the disjoint index partition based on colors from number of pieces */ 
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
    
    
    // We only need to setup the the ghost regions at the top level task
    //  this will ensure that we can setup simulataneous coherence for these
    //  logical regions allowing masters to update ghosts 
    IndexPartition ippslaveghost = runtime->create_index_partition(
      ctx, isp, colorpslaveghost, false);
    lppslaveghost = runtime->get_logical_partition_by_tree(
      ctx, ippslaveghost, fsp_ghost, lrp.get_tree_id());

    IndexPartition ippmasterghost = runtime->create_index_partition(
      ctx, isp, colorpmasterghost, false);
    lppmasterghost = runtime->get_logical_partition_by_tree(
      ctx, ippmasterghost, fsp_ghost, lrp.get_tree_id());


    
#ifdef PENNANT_DEPENDENT_PARTITIONING
    /* note that colorpslaveghost has multicolor points, that is a point may belong to 
       more than one color in this partitioning so it is not disjoint */ 
    IndexPartition ippslaveghost = runtime->create_index_partition(
      ctx, isp, colorpslaveghost, false); 

    /* this pending partition will be used to populate intersections of points
       that is the intersection of points with two colors, master and slave */ 
    IndexPartition ippghosts_individual = runtime->create_pending_partition(ctx, isp, color_space_ghost);
    
    DEBUG("Created individual ghosts using create_pending_partition" << std::endl);
    /* create a ghost region for each pair of colors */
    for (int c0 = 0; c0 < numpcs; ++c0){
      DEBUG("At the top of the create ghost regions loop" << std::endl);
      IndexSpace iss_ghost_c0 = runtime->get_index_subspace(ctx, ippghost, c0);
      DEBUG("Created index space for color " << c0 << std::endl);
      sprintf(buf, "iss_ghost_%d", c0);
      runtime->attach_name(iss_ghost_c0, buf);
      for (int c1 = 0; c1 < numpcs; ++c1) {  // fix me.. this N^2 is not needed //
        if(c1 == c0) continue;
        IndexSpace iss_ghost_c1 = runtime->get_index_subspace(ctx, ippghost, c1);
        DEBUG("Created index space for color " << c1 << std::endl);

        sprintf(buf, "iss_ghost_%d", c1);
        runtime->attach_name(iss_ghost_c1, buf);
        std::vector<IndexSpace> iss_handles;
        iss_handles.push_back(iss_ghost_c0); 
        iss_handles.push_back(iss_ghost_c1);
        /* Compute intersection of points with colors c1 and c0 */
        /* note linearization of two color space... */
        DEBUG("Creating index space for intersection colors  " << c0 << "," << c1 << " aka: " << c0+c1*numpcs  << std::endl);
        DomainPoint dp = DomainPoint::from_point<1>(c0+c1*numpcs);
        IndexSpace iss_ghost_c0_c1 = runtime->create_index_space_intersection(ctx, ippghosts_individual, dp, iss_handles);
        sprintf(buf, "iss_ghost_%d_%d", c0, c1);
        runtime->attach_name(iss_ghost_c0_c1, buf);

        LogicalRegion lr_ghost =runtime->create_logical_region(ctx, iss_ghost_c0_c1, fsp_ghost);
        sprintf(buf, "lr_ghost_%d_%d", c0, c1);
        runtime->attach_name(lr_ghost, buf);
        ghost_regions.push_back(lr_ghost);                                                        
      }
    }
#endif

    // for (int c = 0; c < numpcs; ++c) {
    //   /* only one notifier of ready per color (the master) */ 
    //   args[c].notify_ready[0] = ready_barriers[c];
      
    //   for ( int pt = 0; pt < colorpshr[c].points.size(); ++pt) {
    //     /* master must wait for all slaves to be empty */ 
    //     args[c].wait_empty[pt] = empty_barriers[c];
    //     args[c].notify_empty[0] = empty_barriers[c];
        
    //     //args[c].wait_ready; 
        
    //   }
    // }

    
    
    for(my_map_set::iterator it = map_master_to_slave.begin(); it != map_master_to_slave.end(); it++) {
      for(my_set::iterator its = it->second.begin(); its != it->second.end(); its++){
        auto c0 = it->first;
        auto c1 = *its; 
        DEBUG("Creating master ghost region on colors: " << c0 << "," <<  c1 << std::endl);
        IndexSpace iss_master_ghost = runtime->get_index_subspace(ctx, ippmasterghost,
                                                                  c0+c1*numpcs);
        sprintf(buf, "iss_master_ghost_%d_%d", c0, c1);
        runtime->attach_name(iss_master_ghost, buf);
        LogicalRegion lr_master_ghost =runtime->create_logical_region(ctx, iss_master_ghost, fsp_ghost);
        sprintf(buf, "lr_master_ghost_%d_%d", c0, c1);
        runtime->attach_name(lr_master_ghost, buf);
        master_ghost_regions[c0][c1] = lr_master_ghost;
      }
    }
    
    for(my_map_set::iterator it = map_slaves_to_masters.begin(); it != map_slaves_to_masters.end(); it++) {
      for(my_set::iterator its = it->second.begin(); its != it->second.end(); its++){
        auto c0 = it->first;
        auto c1 = *its; 
        DEBUG("Creating slave ghost region on colors: " << c0 << "," << c1  << std::endl);
        IndexSpace iss_slave_ghost = runtime->get_index_subspace(ctx, ippslaveghost,
                                                                 c0+c1*numpcs);
        sprintf(buf, "iss_slave_ghost_%d_%d", c0, c1);
        runtime->attach_name(iss_slave_ghost, buf);
        LogicalRegion lr_slave_ghost =runtime->create_logical_region(ctx, iss_slave_ghost, fsp_ghost);
        sprintf(buf, "lr_slave_ghost_%d_%d", c0, c1);
        runtime->attach_name(lr_slave_ghost, buf);
        slave_ghost_regions[c0][c1] = lr_slave_ghost;
 
      }
    }
    DEBUG("Done creating index spaces and corresponding logical regions for each ghost" << std::endl);
    MustEpochLauncher must_epoch_launcher;

    std::vector<SPMDArgsSerialized> args_s(numpcs);
    // loop through and setup all my SPMD tasks with proper region requirements for
    //   ghosting 
    for (int color = 0; color < numpcs; ++color){
      SPMDArgs  args; 
      TaskLauncher spmd_launcher(TID_SPMD_TASK, 
                               TaskArgument(&args_s[color], sizeof(SPMDArgsSerialized)));

      int idx=0;
#if 1
      // Add logical regions for private and master points, my zones, and my sides
      spmd_launcher.add_region_requirement(
        RegionRequirement(runtime->get_logical_subregion_by_color(ctx, lppprv, color),
                          READ_WRITE, EXCLUSIVE, lrp));
      spmd_launcher.add_field(idx, FID_PX);
      spmd_launcher.add_field(idx, FID_PXP);
      spmd_launcher.add_field(idx, FID_PX0);
      spmd_launcher.add_field(idx, FID_PU);
      spmd_launcher.add_field(idx, FID_PU0);
      spmd_launcher.add_field(idx, FID_PMASWT);
      spmd_launcher.add_field(idx, FID_PF);
      spmd_launcher.add_field(idx, FID_PAP);

      idx++;
      spmd_launcher.add_region_requirement(
        RegionRequirement(runtime->get_logical_subregion_by_color(ctx, lppmstr, color),
                          READ_WRITE, EXCLUSIVE, lrp));
      spmd_launcher.add_field(idx, FID_PX);
      spmd_launcher.add_field(idx, FID_PXP);
      spmd_launcher.add_field(idx, FID_PX0);
      spmd_launcher.add_field(idx, FID_PU);
      spmd_launcher.add_field(idx, FID_PU0);
      spmd_launcher.add_field(idx, FID_PMASWT);
      spmd_launcher.add_field(idx, FID_PF);
      spmd_launcher.add_field(idx, FID_PAP);
      idx++;
      
      spmd_launcher.add_region_requirement(
        RegionRequirement(runtime->get_logical_subregion_by_color(ctx, lps, color),
                          READ_WRITE, EXCLUSIVE, lrs));
      spmd_launcher.add_field(idx, FID_MAPSP1);
      spmd_launcher.add_field(idx, FID_MAPSP2);
      spmd_launcher.add_field(idx, FID_MAPSZ);
      spmd_launcher.add_field(idx, FID_MAPSS3);
      spmd_launcher.add_field(idx, FID_MAPSS4);
      spmd_launcher.add_field(idx, FID_MAPSP1REG);
      spmd_launcher.add_field(idx, FID_MAPSP2REG);
      spmd_launcher.add_field(idx, FID_EX);
      spmd_launcher.add_field(idx, FID_EXP);
      spmd_launcher.add_field(idx, FID_SAREA);
      spmd_launcher.add_field(idx, FID_SVOL);
      spmd_launcher.add_field(idx, FID_SAREAP);
      spmd_launcher.add_field(idx, FID_SVOLP);
      spmd_launcher.add_field(idx, FID_SSURFP);
      spmd_launcher.add_field(idx, FID_ELEN);
      spmd_launcher.add_field(idx, FID_SMF);
      spmd_launcher.add_field(idx, FID_SFP);
      spmd_launcher.add_field(idx, FID_SFQ);
      spmd_launcher.add_field(idx, FID_SFT);
      spmd_launcher.add_field(idx, FID_CAREA);
      spmd_launcher.add_field(idx, FID_CEVOL);
      spmd_launcher.add_field(idx, FID_CDU);
      spmd_launcher.add_field(idx, FID_CDIV);
      spmd_launcher.add_field(idx, FID_CCOS);
      spmd_launcher.add_field(idx, FID_CQE1);
      spmd_launcher.add_field(idx, FID_CQE2);
      spmd_launcher.add_field(idx, FID_CRMU);
      spmd_launcher.add_field(idx, FID_CW);
      idx++;
      
      
      spmd_launcher.add_region_requirement(
        RegionRequirement(runtime->get_logical_subregion_by_color(ctx, lpz, color),
                          READ_WRITE, EXCLUSIVE, lrz));
      spmd_launcher.add_field(idx, FID_ZNUMP);
      spmd_launcher.add_field(idx, FID_ZX);
      spmd_launcher.add_field(idx, FID_ZXP);
      spmd_launcher.add_field(idx, FID_ZAREA);
      spmd_launcher.add_field(idx, FID_ZVOL);
      spmd_launcher.add_field(idx, FID_ZAREAP);
      spmd_launcher.add_field(idx, FID_ZVOLP);
      spmd_launcher.add_field(idx, FID_ZVOL0);
      spmd_launcher.add_field(idx, FID_ZDL);
      spmd_launcher.add_field(idx, FID_ZM);
      spmd_launcher.add_field(idx, FID_ZR);
      spmd_launcher.add_field(idx, FID_ZRP);
      spmd_launcher.add_field(idx, FID_ZE);
      spmd_launcher.add_field(idx, FID_ZETOT);
      spmd_launcher.add_field(idx, FID_ZW);
      spmd_launcher.add_field(idx, FID_ZWRATE);
      spmd_launcher.add_field(idx, FID_ZP);
      spmd_launcher.add_field(idx, FID_ZSS);
      spmd_launcher.add_field(idx, FID_ZDU);
      spmd_launcher.add_field(idx, FID_ZUC);
      spmd_launcher.add_field(idx, FID_ZTMP);
      idx++;
      
      // loop through all ghost regions for which color is master and add them
      //  to the region requirements of the spmd task with READ_WRITE perms
      //  need to start at idx 4 as we addthe other (0-3rd) region requirements above
      //  that way we add the fields to the right region requirement 
#endif
#if 1  
      for(std::map<int, LegionRuntime::HighLevel::LogicalRegion>::iterator its = master_ghost_regions[color].begin(); its != master_ghost_regions[color].end(); ++its) {
        spmd_launcher.add_region_requirement(
          RegionRequirement(its->second, READ_WRITE,
                            SIMULTANEOUS, its->second));
        DEBUG("Adding master ghost to region requirements with colors: " << color << ":" <<
              its->first << std::endl);
        spmd_launcher.add_field(idx, FID_PXP_GHOST);
        spmd_launcher.add_field(idx, FID_PU_GHOST);
        spmd_launcher.add_field(idx, FID_PU0_GHOST);
        spmd_launcher.add_field(idx, FID_PMASWT_GHOST);
        spmd_launcher.add_field(idx, FID_PF_GHOST);
        
        // I'm the master, need to notify my slaves 
        args.notify_ready[its->first] = ready_barriers[color][its->first];
        //  Master must wait on slave to empty ghosts 
        args.wait_empty[its->first] = empty_barriers[color][its->first];
        idx++;
      }
      // now for ghost regions for which color is slave but with READ_ONLY 
      for(std::map<int, LegionRuntime::HighLevel::LogicalRegion>::iterator its = slave_ghost_regions[color].begin(); its != slave_ghost_regions[color].end(); its++) {
        spmd_launcher.add_region_requirement(
          RegionRequirement(its->second, READ_ONLY,
                            SIMULTANEOUS, its->second));
        DEBUG("Adding slave ghost to region requirements with colors: " << color << ":" <<
              its->first << std::endl);
     
        spmd_launcher.add_field(idx, FID_PXP_GHOST);
        spmd_launcher.add_field(idx, FID_PU_GHOST);
        spmd_launcher.add_field(idx, FID_PU0_GHOST);
        spmd_launcher.add_field(idx, FID_PMASWT_GHOST);
        spmd_launcher.add_field(idx, FID_PF_GHOST);
        // I'm a slave waiting on master for point updates 
        args.wait_ready[its->first] = ready_barriers[its->first][color];
        // Master is waiting on me to empty my ghosts 
        args.notify_empty[its->first] = empty_barriers[its->first][color];
        
        idx++;
      }
#endif
      Realm::Serialization::DynamicBufferSerializer dbs(0);
      dbs << args;
      args_s[color].my_size = dbs.bytes_used();
      memcpy(args_s[color].my_data, dbs.detach_buffer(), args_s[color].my_size);
      
      spmd_launcher.add_index_requirement(IndexSpaceRequirement(isp, NO_MEMORY, isp));
      
      DomainPoint point(color);
      must_epoch_launcher.add_single_task(point, spmd_launcher);
      DEBUG("Finished setting up SPMD task for color: " << color
            << " arg size is: " << args_s[color].my_size
            << " number of region requirements is: " << spmd_launcher.region_requirements.size()
            << std::endl);
      
    }
    DEBUG("Launching SPMD tasks using must epoch" << std::endl);
    FutureMap fm = runtime->execute_must_epoch(ctx, must_epoch_launcher);
    fm.wait_all_results();
    DEBUG("SPMD tasks complete" << std::endl);
    
    vector<ptr_t> lgmapsp1(&mapsp1[0], &mapsp1[nums]);
    vector<ptr_t> lgmapsp2(&mapsp2[0], &mapsp2[nums]);
    vector<ptr_t> lgmapsz (&mapsz [0], &mapsz [nums]);
    vector<ptr_t> lgmapss3(&mapss3[0], &mapss3[nums]);
    vector<ptr_t> lgmapss4(&mapss4[0], &mapss4[nums]);

    vector<int> lgmapsp1reg(nums), lgmapsp2reg(nums);
    for (int s = 0; s < nums; ++s) {
        lgmapsp1reg[s] = (pointcolors[mapsp1[s]] == MULTICOLOR);
        lgmapsp2reg[s] = (pointcolors[mapsp2[s]] == MULTICOLOR);
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
        std::vector<int>& zonestart,
        std::vector<int>& zonesize,
        std::vector<int>& zonenodes) {

    mapsp1 = alloc<int>(nums);
    mapsp2 = alloc<int>(nums);
    mapsz  = alloc<int>(nums);
    mapss3 = alloc<int>(nums);
    mapss4 = alloc<int>(nums);

    for (int z = 0; z < numz; ++z) {
        int sbase = zonestart[z];
        int size = zonesize[z];
        for (int n = 0; n < size; ++n) {
            int s = sbase + n;
            int snext = sbase + (n + 1 == size ? 0 : n + 1);
            int slast = sbase + (n == 0 ? size : n) - 1;
            mapsz[s] = z;
            mapsp1[s] = zonenodes[s];
            mapsp2[s] = zonenodes[snext];
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

void Mesh::stencilTask(
  const Task *task,
  const std::vector<PhysicalRegion> & regions,
  Context ctx,
  HighLevelRuntime *runtime) {
  
  
}

void Mesh::SPMDtask(
  const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx,
  HighLevelRuntime *runtime) {
  // Unmap all the regions we were given since, we don't use them in this task
  runtime->unmap_all_regions(ctx);
  SPMDArgsSerialized *serial_args = (SPMDArgsSerialized*) task->args;
  Realm::Serialization::FixedBufferDeserializer fdb(serial_args->my_data, serial_args->my_size);
  SPMDArgs args;
  bool ok = fdb >> args;
  if(!ok) {
    std::cerr << "ERROR in SPMDTask, can't deserialize!" << std::endl;
  } else {
    std::cout << "In SPMDTask with notify_ready: " << args.notify_ready.size()
          << " notify_empty: " << args.notify_empty.size()
          << " wait_ready: " << args.wait_ready.size()
          << " wait_empty: " << args.wait_empty.size() << std::endl;
  }

  /* now let's pull out our logical regions that will be needed for the stencil */
  LogicalRegion lrp_private = task->regions[0].region; 
  LogicalRegion lrp_master = task->regions[1].region;
  LogicalRegion lrs = task->regions[2].region;
  LogicalRegion lrz = task->regions[3].region;
  
//  LogicalRegion& lrglb = ; 
    
  
  
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
    //LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::SOA<sizeof(T)> , T> acc_src;
    //acc_src = regions[0].get_field_accessor(fid_src).typeify<T>(). template convert<LegionRuntime::Accessor::AccessorType::SOA<sizeof(T)> >(); 
      
    MyAccessor<T> acc_src = get_accessor<T>(regions[0], fid_src); //regions[0].get_field_accessor(fid_src).typeify<T>().convert<AccessorType::SOA>();
    //LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::SOA<sizeof(T)> , T> acc_dst;
    //acc_dst = regions[1].get_field_accessor(fid_dst).typeify<T>(). template convert<LegionRuntime::Accessor::AccessorType::SOA<sizeof(T)> >(); 
    MyAccessor<T> acc_dst =
      get_accessor<T>(regions[1], fid_dst);

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

    #pragma ivdep
    for (int s = sfirst; s < slast; ++s) {
        int z = mapsz[s];
        smf[s] = sarea[s] / zarea[z];
    }
}



