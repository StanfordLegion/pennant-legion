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
using namespace LegionRuntime::Accessor;


namespace {  // unnamed
static void __attribute__ ((constructor)) registerTasks() {
    TaskVariantRegistrar registrar(TID_APPLYFIXEDBC, "applyfixedbc");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<HydroBC::applyFixedBCTask>(registrar);
}
}; // namespace


HydroBC::HydroBC(
        Mesh* msh,
        const double2 v,
        const vector<int>& mbp)
    : mesh(msh), numb(mbp.size()), vfix(v) {

    mapbp = alloc<int>(numb);
    copy(mbp.begin(), mbp.end(), mapbp);

    mesh->getPlaneChunks(numb, mapbp, pchbfirst, pchblast);

    Context ctx = mesh->ctx;
    Runtime* runtime = mesh->runtime;

    // create index space for boundary points
    IndexSpace isb = runtime->create_index_space(ctx, numb);
    IndexAllocator iab = runtime->create_index_allocator(ctx, isb);
    iab.alloc(numb);
    FieldSpace fsb = runtime->create_field_space(ctx);
    FieldAllocator fab = runtime->create_field_allocator(ctx, fsb);
    fab.allocate_field(sizeof(ptr_t), FID_MAPBP);
    fab.allocate_field(sizeof(int), FID_MAPBPREG);
    lrb = runtime->create_logical_region(ctx, isb, fsb);

    // create boundary point partition
    Coloring colorb;
    // force all colors to exist, even if they might be empty
    for (int c = 0; c < mesh->numpcs; ++c) {
        colorb[c];
    }
    for (int b = 0; b < numb; ++b) {
        int p = mbp[b];
        int c = mesh->nodecolors[p];
        if (c == MULTICOLOR) c = mesh->nodemcolors[p][0];
        colorb[c].points.insert(b);
    }
    IndexPartition ipb = runtime->create_index_partition(
                ctx, isb, colorb, true);
    lpb = runtime->get_logical_partition(ctx, lrb, ipb);

    // create boundary point maps
    vector<ptr_t> lgmapbp(&mbp[0], &mbp[numb]);
    vector<int> lgmapbpreg(numb);
    for (int b = 0; b < numb; ++b) {
        lgmapbpreg[b] = (mesh->nodecolors[mbp[b]] == MULTICOLOR);
    }

    mesh->setField(lrb, FID_MAPBP, &lgmapbp[0], numb);
    mesh->setField(lrb, FID_MAPBPREG, &lgmapbpreg[0], numb);

}


HydroBC::~HydroBC() {}


void HydroBC::applyFixedBCTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double2* args = (const double2*) task->args;
    const double2 vfix = args[0];
    MyAccessor<ptr_t> acc_mapbp =
        get_accessor<ptr_t>(regions[0], FID_MAPBP);
    MyAccessor<int> acc_mapbpreg =
        get_accessor<int>(regions[0], FID_MAPBPREG);
    MyAccessor<double2> acc_pf[2] = {
        get_accessor<double2>(regions[1], FID_PF),
        get_accessor<double2>(regions[2], FID_PF)
    };
    MyAccessor<double2> acc_pu[2] = {
        get_accessor<double2>(regions[1], FID_PU0),
        get_accessor<double2>(regions[2], FID_PU0)
    };

    const IndexSpace& isb = task->regions[0].region.get_index_space();
   
    for (IndexIterator itrb(runtime, ctx, isb); itrb.has_next(); )
    {
        ptr_t b = itrb.next();
        ptr_t p = acc_mapbp.read(b);
        int preg = acc_mapbpreg.read(b);
        double2 pu = acc_pu[preg].read(p);
        double2 pf = acc_pf[preg].read(p);
        pu = project(pu, vfix);
        pf = project(pf, vfix);
        acc_pu[preg].write(p, pu);
        acc_pf[preg].write(p, pf);
    }

}

