/*
 * Hydro.cc
 *
 *  Created on: Dec 22, 2011
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "Hydro.hh"

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

#include "legion.h"

#include "MyLegion.hh"
#include "Memory.hh"
#include "InputFile.hh"
#include "Mesh.hh"
#include "PolyGas.hh"
#include "TTS.hh"
#include "QCS.hh"
#include "HydroBC.hh"

using namespace std;
using namespace Memory;
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


namespace {  // unnamed
static void __attribute__ ((constructor)) registerTasks() {
  
    HighLevelRuntime::register_legion_task<Hydro::advPosHalfTask>(
            TID_ADVPOSHALF, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "advposhalf");
    HighLevelRuntime::register_legion_task<Hydro::calcRhoTask>(
            TID_CALCRHO, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "calcrho");
    HighLevelRuntime::register_legion_task<Hydro::calcCrnrMassTask>( // this updates ghosts? 
            TID_CALCCRNRMASS, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "calccrnrmass");
    HighLevelRuntime::register_legion_task<Hydro::sumCrnrForceTask>(
            TID_SUMCRNRFORCE, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "sumcrnrforce");
    HighLevelRuntime::register_legion_task<Hydro::calcAccelTask>(
            TID_CALCACCEL, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "calcaccel");
    HighLevelRuntime::register_legion_task<Hydro::advPosFullTask>(
            TID_ADVPOSFULL, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "advposfull");
    HighLevelRuntime::register_legion_task<Hydro::calcWorkTask>(
            TID_CALCWORK, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "calcwork");
    HighLevelRuntime::register_legion_task<Hydro::calcWorkRateTask>(
            TID_CALCWORKRATE, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "calcworkrate");
    HighLevelRuntime::register_legion_task<Hydro::calcEnergyTask>(
            TID_CALCENERGY, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "calcenergy");
    HighLevelRuntime::register_legion_task<double, Hydro::calcDtTask>(
            TID_CALCDT, Processor::LOC_PROC, true, true,
            AUTO_GENERATE_ID, TaskConfigOptions(true), "calcdt");
}
}; // namespace


Hydro::Hydro(
        const InputFile* inp,
        Mesh* m,
        Context ctxa,
        HighLevelRuntime* runtimea)
        : mesh(m), ctx(ctxa), runtime(runtimea) {
    cfl = inp->getDouble("cfl", 0.6);
    cflv = inp->getDouble("cflv", 0.1);
    rinit = inp->getDouble("rinit", 1.);
    einit = inp->getDouble("einit", 0.);
    rinitsub = inp->getDouble("rinitsub", 1.);
    einitsub = inp->getDouble("einitsub", 0.);
    uinitradial = inp->getDouble("uinitradial", 0.);
    bcx = inp->getDoubleList("bcx", vector<double>());
    bcy = inp->getDoubleList("bcy", vector<double>());

    pgas = new PolyGas(inp, this);
    tts = new TTS(inp, this);
    qcs = new QCS(inp, this);

    const double2 vfixx = double2(1., 0.);
    const double2 vfixy = double2(0., 1.);
    for (int i = 0; i < bcx.size(); ++i)
        bcs.push_back(new HydroBC(mesh, vfixx, mesh->getXPlane(bcx[i])));
    for (int i = 0; i < bcy.size(); ++i)
        bcs.push_back(new HydroBC(mesh, vfixy, mesh->getYPlane(bcy[i])));

    init();
}


Hydro::~Hydro() {

    delete tts;
    delete qcs;
    for (int i = 0; i < bcs.size(); ++i) {
        delete bcs[i];
    }
}


void Hydro::init() {

    const int numpch = mesh->numpch;
    const int numzch = mesh->numzch;
    const int nump = mesh->nump;
    const int numz = mesh->numz;
    const int nums = mesh->nums;

    const double2* zx = mesh->zx;
    const double* zvol = mesh->zvol;

    // allocate arrays
    pu = alloc<double2>(nump);
    zm = alloc<double>(numz);
    zr = alloc<double>(numz);
    ze = alloc<double>(numz);
    zetot = alloc<double>(numz);
    zwrate = alloc<double>(numz);
    zp = alloc<double>(numz);

    // initialize hydro vars
    for (int zch = 0; zch < numzch; ++zch) {
        int zfirst = mesh->zchzfirst[zch];
        int zlast = mesh->zchzlast[zch];

        fill(&zr[zfirst], &zr[zlast], rinit);
        fill(&ze[zfirst], &ze[zlast], einit);
        fill(&zwrate[zfirst], &zwrate[zlast], 0.);

        const vector<double>& subrgn = mesh->subregion;
        if (!subrgn.empty()) {
            const double eps = 1.e-12;
            #pragma ivdep
            for (int z = zfirst; z < zlast; ++z) {
                if (zx[z].x > (subrgn[0] - eps) &&
                    zx[z].x < (subrgn[1] + eps) &&
                    zx[z].y > (subrgn[2] - eps) &&
                    zx[z].y < (subrgn[3] + eps)) {
                    zr[z] = rinitsub;
                    ze[z] = einitsub;
                }
            }
        }

        #pragma ivdep
        for (int z = zfirst; z < zlast; ++z) {
            zm[z] = zr[z] * zvol[z];
            zetot[z] = ze[z] * zm[z];
        }
    }  // for sch

    for (int pch = 0; pch < numpch; ++pch) {
        int pfirst = mesh->pchpfirst[pch];
        int plast = mesh->pchplast[pch];
        if (uinitradial != 0.)
            initRadialVel(uinitradial, pfirst, plast);
        else
            fill(&pu[pfirst], &pu[plast], double2(0., 0.));
    }  // for pch

    // FIXME: This doesn't work, the fields won't appear in this tasks
    // region requirements.
    // FieldSpace fsp = mesh->lrp.get_field_space();
    // FieldAllocator fap = runtime->create_field_allocator(ctx, fsp);
    // fap.allocate_field(sizeof(double2), FID_PU);
    // fap.allocate_field(sizeof(double2), FID_PU0);
    // fap.allocate_field(sizeof(double), FID_PMASWT);
    // fap.allocate_field(sizeof(double2), FID_PF);
    // fap.allocate_field(sizeof(double2), FID_PAP);

    // FieldSpace fsz = mesh->lrz.get_field_space();
    // FieldAllocator faz = runtime->create_field_allocator(ctx, fsz);
    // faz.allocate_field(sizeof(double), FID_ZM);
    // faz.allocate_field(sizeof(double), FID_ZR);
    // faz.allocate_field(sizeof(double), FID_ZRP);
    // faz.allocate_field(sizeof(double), FID_ZE);
    // faz.allocate_field(sizeof(double), FID_ZETOT);
    // faz.allocate_field(sizeof(double), FID_ZW);
    // faz.allocate_field(sizeof(double), FID_ZWRATE);
    // faz.allocate_field(sizeof(double), FID_ZP);
    // faz.allocate_field(sizeof(double), FID_ZSS);
    // faz.allocate_field(sizeof(double), FID_ZDU);

    // FieldSpace fss = mesh->lrs.get_field_space();
    // FieldAllocator fas = runtime->create_field_allocator(ctx, fss);
    // fas.allocate_field(sizeof(double2), FID_SFP);
    // fas.allocate_field(sizeof(double2), FID_SFQ);
    // fas.allocate_field(sizeof(double2), FID_SFT);

    // FieldSpace fsglb = mesh->lrglb.get_field_space();
    // FieldAllocator faglb = runtime->create_field_allocator(ctx, fsglb);
    // faglb.allocate_field(sizeof(double), FID_DTREC);

    LogicalRegion& lrp = mesh->lrp;
    LogicalRegion& lrz = mesh->lrz;
    mesh->setField(lrp, FID_PU, pu, nump);
    mesh->setField(lrz, FID_ZM, zm, numz);
    mesh->setField(lrz, FID_ZR, zr, numz);
    mesh->setField(lrz, FID_ZE, ze, numz);
    mesh->setField(lrz, FID_ZETOT, zetot, numz);
    mesh->setField(lrz, FID_ZWRATE, zwrate, numz);

}


void Hydro::initRadialVel(
        const double vel,
        const int pfirst,
        const int plast) {
    const double2* px = mesh->px;
    const double eps = 1.e-12;

    #pragma ivdep
    for (int p = pfirst; p < plast; ++p) {
        double pmag = length(px[p]);
        if (pmag > eps)
            pu[p] = vel * px[p] / pmag;
        else
            pu[p] = double2(0., 0.);
    }
}


void Hydro::doCycle(
            const double dt) {

    resetDtHydro();

    LogicalRegion& lrp = mesh->lrp;
    LogicalRegion& lrs = mesh->lrs;
    LogicalRegion& lrz = mesh->lrz;
    LogicalPartition& lppprv = mesh->lppprv;
    LogicalPartition& lppmstr = mesh->lppmstr;
    LogicalPartition& lppshr = mesh->lppshr;
    LogicalPartition& lps = mesh->lps;
    LogicalPartition& lpz = mesh->lpz;
    LogicalRegion& lrglb = mesh->lrglb;
    Domain& dompc = mesh->dompc;

    TaskArgument ta;
    ArgumentMap am;

    runtime->begin_trace(ctx, 123);
#if 0
    // store fields from last cycle where needed
    CopyLauncher launchcfd;
    launchcfd.add_copy_requirements(
      RegionRequirement(lrz, READ_ONLY, EXCLUSIVE, lrz), 
      RegionRequirement(lrz, WRITE_DISCARD, EXCLUSIVE, lrz));
    
    launchcfd.add_src_field(0, FID_ZVOL);
    launchcfd.add_dst_field(0, FID_ZVOL0);
    
    runtime->issue_copy_operation(ctx, launchcfd);
#endif
#if 1 
    IndexLauncher launchcfd(TID_COPYFIELDDBL, dompc, ta, am);
    launchcfd.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcfd.add_field(0, FID_ZVOL);
    launchcfd.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchcfd.add_field(1, FID_ZVOL0);
    runtime->execute_index_space(ctx, launchcfd);
    
#endif
    
    // begin hydro cycle
    IndexLauncher launchcfd2(TID_COPYFIELDDBL2, dompc, ta, am);
    double ffdargs[] = { 0. };
    IndexLauncher launchffd(TID_FILLFIELDDBL, dompc,
            TaskArgument(ffdargs, sizeof(ffdargs)), am);
    double2 ffd2args[] = { double2(0., 0.) };
    IndexLauncher launchffd2(TID_FILLFIELDDBL2, dompc,
            TaskArgument(ffd2args, sizeof(ffd2args)), am);
    double aphargs[] = { dt };
    IndexLauncher launchaph(TID_ADVPOSHALF, dompc,
                            TaskArgument(aphargs, sizeof(aphargs)), am, Predicate::TRUE_PRED,
                            false);
    // do point routines twice, once each for private and master
    // partitions
    for (int part = 0; part < 2; ++part) {
        LogicalPartition& lppcurr = (part == 0 ? lppprv : lppmstr);
        launchcfd2.region_requirements.clear();
        launchcfd2.add_region_requirement(
                RegionRequirement(lppcurr, 0,
                        READ_ONLY, EXCLUSIVE, lrp));
        launchcfd2.add_field(0, FID_PX);
        launchcfd2.add_region_requirement(
                RegionRequirement(lppcurr, 0,
                        WRITE_DISCARD, EXCLUSIVE, lrp));
        launchcfd2.add_field(1, FID_PX0);
        // save off point variable values from previous step (FID_PX -> FID_PX0)
        //   - no shared point updates to slave.. 
        runtime->execute_index_space(ctx, launchcfd2);

        // reuse copy launcher for different field
        launchcfd2.region_requirements[0].privilege_fields.clear();
        launchcfd2.region_requirements[0].instance_fields.clear();
        launchcfd2.add_field(0, FID_PU);
        launchcfd2.region_requirements[1].privilege_fields.clear();
        launchcfd2.region_requirements[1].instance_fields.clear();
        launchcfd2.add_field(1, FID_PU0);
        // save off point variable values from previous step (FID_PU -> FID_PU0)
        //   - no shared point updates to slave.. 
        runtime->execute_index_space(ctx, launchcfd2);

        launchffd.region_requirements.clear();
        launchffd.add_region_requirement(
                RegionRequirement(lppcurr, 0,
                        WRITE_DISCARD, EXCLUSIVE, lrp));
        launchffd.add_field(0, FID_PMASWT);

        // fill point (FID_PMASWT) with 0..
        runtime->execute_index_space(ctx, launchffd);

        
        launchffd2.region_requirements.clear();
        launchffd2.add_region_requirement(
                RegionRequirement(lppcurr, 0,
                        WRITE_DISCARD, EXCLUSIVE, lrp));
        launchffd2.add_field(0, FID_PF);

        // fill point (FID_PF) with 0..
        runtime->execute_index_space(ctx, launchffd2);

        launchaph.region_requirements.clear();
        launchaph.add_region_requirement(
                RegionRequirement(lppcurr, 0,
                        READ_ONLY, EXCLUSIVE, lrp));
        launchaph.add_field(0, FID_PX0);
        launchaph.add_field(0, FID_PU0);
        launchaph.add_region_requirement(
                RegionRequirement(lppcurr, 0,
                        WRITE_DISCARD, EXCLUSIVE, lrp));
        launchaph.add_field(1, FID_PXP);
        // ======= Predictor step ======
        // 1. Advance mesh to center of time step
        //  updates master and private points, note that the tasks below
        //  will access ghost versions of these points, so an update of ghosts is required here. 
        runtime->execute_index_space(ctx, launchaph);
    }  // for part

    // 1.a Computer new mesh geometry 
    IndexLauncher launchcc(TID_CALCCTRS, dompc, ta, am);
    launchcc.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchcc.add_field(0, FID_MAPSP1);
    launchcc.add_field(0, FID_MAPSP2);
    launchcc.add_field(0, FID_MAPSZ);
    launchcc.add_field(0, FID_MAPSP1REG);
    launchcc.add_field(0, FID_MAPSP2REG);
    launchcc.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcc.add_field(1, FID_ZNUMP);
    launchcc.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchcc.add_field(2, FID_PXP);
    launchcc.add_region_requirement(
            RegionRequirement(lppshr, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchcc.add_field(3, FID_PXP);
    launchcc.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchcc.add_field(4, FID_EXP);
    launchcc.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchcc.add_field(5, FID_ZXP);
    // requires read access to ghost points, updates private zones and sides 
    runtime->execute_index_space(ctx, launchcc);

    IndexLauncher launchcv(TID_CALCVOLS, dompc, ta, am);
    launchcv.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchcv.add_field(0, FID_MAPSP1);
    launchcv.add_field(0, FID_MAPSP2);
    launchcv.add_field(0, FID_MAPSZ);
    launchcv.add_field(0, FID_MAPSP1REG);
    launchcv.add_field(0, FID_MAPSP2REG);
    launchcv.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchcv.add_field(1, FID_PXP);
    launchcv.add_region_requirement(
            RegionRequirement(lppshr, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchcv.add_field(2, FID_PXP);
    launchcv.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcv.add_field(3, FID_ZXP);
    launchcv.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchcv.add_field(4, FID_SAREAP);
    launchcv.add_field(4, FID_SVOLP);
    launchcv.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchcv.add_field(5, FID_ZAREAP);
    launchcv.add_field(5, FID_ZVOLP);
    // requires read access to ghost points, updates private zones and sides 
    mesh->fmapcv = runtime->execute_index_space(ctx, launchcv);

    IndexLauncher launchcsv(TID_CALCSURFVECS, dompc, ta, am);
    launchcsv.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchcsv.add_field(0, FID_MAPSZ);
    launchcsv.add_field(0, FID_EXP);
    launchcsv.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcsv.add_field(1, FID_ZXP);
    launchcsv.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchcsv.add_field(2, FID_SSURFP);
    // no ghost access required (all private partitions) 
    runtime->execute_index_space(ctx, launchcsv);

    IndexLauncher launchcel(TID_CALCEDGELEN, dompc, ta, am);
    launchcel.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchcel.add_field(0, FID_MAPSP1);
    launchcel.add_field(0, FID_MAPSP2);
    launchcel.add_field(0, FID_MAPSP1REG);
    launchcel.add_field(0, FID_MAPSP2REG);
    launchcel.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchcel.add_field(1, FID_PXP);
    launchcel.add_region_requirement(
            RegionRequirement(lppshr, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchcel.add_field(2, FID_PXP);
    launchcel.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchcel.add_field(3, FID_ELEN);
    // requires read access to ghost points and write access to private edges 
    runtime->execute_index_space(ctx, launchcel);

    IndexLauncher launchccl(TID_CALCCHARLEN, dompc, ta, am);
    launchccl.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchccl.add_field(0, FID_MAPSZ);
    launchccl.add_field(0, FID_SAREAP);
    launchccl.add_field(0, FID_ELEN);
    launchccl.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchccl.add_field(1, FID_ZNUMP);
    launchccl.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchccl.add_field(2, FID_ZDL);
    // no ghost access required (all private partitions) 
    runtime->execute_index_space(ctx, launchccl);

    // 2. compute point masses 
    IndexLauncher launchcr(TID_CALCRHO, dompc, ta, am);
    launchcr.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcr.add_field(0, FID_ZM);
    launchcr.add_field(0, FID_ZVOLP);
    launchcr.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchcr.add_field(1, FID_ZRP);
    // no ghost access required (all private partitions) 
    runtime->execute_index_space(ctx, launchcr);
    
    IndexLauncher launchccm(TID_CALCCRNRMASS, dompc, ta, am);
    launchccm.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchccm.add_field(0, FID_MAPSP1);
    launchccm.add_field(0, FID_MAPSP1REG);
    launchccm.add_field(0, FID_MAPSS3);
    launchccm.add_field(0, FID_MAPSZ);
    launchccm.add_field(0, FID_SMF);
    launchccm.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchccm.add_field(1, FID_ZRP);
    launchccm.add_field(1, FID_ZAREAP);
    launchccm.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_WRITE, EXCLUSIVE, lrp));
    launchccm.add_field(2, FID_PMASWT);
    launchccm.add_region_requirement(
            RegionRequirement(lppshr, 0, OPID_SUMDBL,
                    SIMULTANEOUS, lrp));
    launchccm.add_field(3, FID_PMASWT);
    // requires read / write access to private and shared (ghosted) points
    //   uses a reduction operation to sum corner masses to point mass 
    runtime->execute_index_space(ctx, launchccm);

    // 3. compute material state (half-advanced) 
    double cshargs[] = { pgas->gamma, pgas->ssmin, dt };
    IndexLauncher launchcsh(TID_CALCSTATEHALF, dompc,
            TaskArgument(cshargs, sizeof(cshargs)), am);
    launchcsh.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcsh.add_field(0, FID_ZR);
    launchcsh.add_field(0, FID_ZVOLP);
    launchcsh.add_field(0, FID_ZVOL0);
    launchcsh.add_field(0, FID_ZE);
    launchcsh.add_field(0, FID_ZWRATE);
    launchcsh.add_field(0, FID_ZM);
    launchcsh.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchcsh.add_field(1, FID_ZP);
    launchcsh.add_field(1, FID_ZSS);
    // no ghost access required (all private partitions) 
    runtime->execute_index_space(ctx, launchcsh);

    // 4. compute forces 
    IndexLauncher launchcfp(TID_CALCFORCEPGAS, dompc, ta, am);
    launchcfp.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchcfp.add_field(0, FID_MAPSZ);
    launchcfp.add_field(0, FID_SSURFP);
    launchcfp.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcfp.add_field(1, FID_ZP);
    launchcfp.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchcfp.add_field(2, FID_SFP);
    // no ghost access required (all private partitions) 
    runtime->execute_index_space(ctx, launchcfp);

    double cftargs[] = { tts->alfa, tts->ssmin };
    IndexLauncher launchcft(TID_CALCFORCETTS, dompc,
            TaskArgument(cftargs, sizeof(cftargs)), am);
    launchcft.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchcft.add_field(0, FID_MAPSZ);
    launchcft.add_field(0, FID_SAREAP);
    launchcft.add_field(0, FID_SMF);
    launchcft.add_field(0, FID_SSURFP);
    launchcft.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcft.add_field(1, FID_ZAREAP);
    launchcft.add_field(1, FID_ZRP);
    launchcft.add_field(1, FID_ZSS);
    launchcft.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchcft.add_field(2, FID_SFT);
    // no ghost access required (all private partitions) 
    runtime->execute_index_space(ctx, launchcft);

    IndexLauncher launchscd(TID_SETCORNERDIV, dompc, ta, am);
    launchscd.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchscd.add_field(0, FID_MAPSZ);
    launchscd.add_field(0, FID_MAPSP1);
    launchscd.add_field(0, FID_MAPSP2);
    launchscd.add_field(0, FID_MAPSS3);
    launchscd.add_field(0, FID_MAPSP1REG);
    launchscd.add_field(0, FID_MAPSP2REG);
    launchscd.add_field(0, FID_EXP);
    launchscd.add_field(0, FID_ELEN);
    launchscd.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchscd.add_field(1, FID_ZNUMP);
    launchscd.add_field(1, FID_ZXP);
    launchscd.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchscd.add_field(2, FID_PXP);
    launchscd.add_field(2, FID_PU0);
    launchscd.add_region_requirement(
            RegionRequirement(lppshr, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchscd.add_field(3, FID_PXP);
    launchscd.add_field(3, FID_PU0);
    launchscd.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchscd.add_field(4, FID_ZUC);
    launchscd.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchscd.add_field(5, FID_CAREA);
    launchscd.add_field(5, FID_CCOS);
    launchscd.add_field(5, FID_CDIV);
    launchscd.add_field(5, FID_CEVOL);
    launchscd.add_field(5, FID_CDU);
    // requires read only access to shared (ghost) and private point data 
    runtime->execute_index_space(ctx, launchscd);

    double sqcfargs[] = { qcs->qgamma, qcs->q1, qcs->q2 };
    IndexLauncher launchsqcf(TID_SETQCNFORCE, dompc,
            TaskArgument(sqcfargs, sizeof(sqcfargs)), am);
    launchsqcf.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchsqcf.add_field(0, FID_MAPSZ);
    launchsqcf.add_field(0, FID_MAPSP1);
    launchsqcf.add_field(0, FID_MAPSP2);
    launchsqcf.add_field(0, FID_MAPSS3);
    launchsqcf.add_field(0, FID_MAPSP1REG);
    launchsqcf.add_field(0, FID_MAPSP2REG);
    launchsqcf.add_field(0, FID_ELEN);
    launchsqcf.add_field(0, FID_CDIV);
    launchsqcf.add_field(0, FID_CDU);
    launchsqcf.add_field(0, FID_CEVOL);
    launchsqcf.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchsqcf.add_field(1, FID_ZRP);
    launchsqcf.add_field(1, FID_ZSS);
    launchsqcf.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchsqcf.add_field(2, FID_PU0);
    launchsqcf.add_region_requirement(
            RegionRequirement(lppshr, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchsqcf.add_field(3, FID_PU0);
    launchsqcf.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchsqcf.add_field(4, FID_CRMU);
    launchsqcf.add_field(4, FID_CQE1);
    launchsqcf.add_field(4, FID_CQE2);
    // requires read only access to shared (ghost) and private point data 
    runtime->execute_index_space(ctx, launchsqcf);

    IndexLauncher launchsfq(TID_SETFORCEQCS, dompc, ta, am);
    launchsfq.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchsfq.add_field(0, FID_MAPSS4);
    launchsfq.add_field(0, FID_CAREA);
    launchsfq.add_field(0, FID_CQE1);
    launchsfq.add_field(0, FID_CQE2);
    launchsfq.add_field(0, FID_ELEN);
    launchsfq.add_region_requirement(
            RegionRequirement(lps, 0, READ_WRITE, EXCLUSIVE, lrs));
    launchsfq.add_field(1, FID_CCOS);
    launchsfq.add_region_requirement(
            RegionRequirement(lps, 0, WRITE_DISCARD, EXCLUSIVE, lrs));
    launchsfq.add_field(2, FID_CW);
    launchsfq.add_field(2, FID_SFQ);
    // no ghost data required 
    runtime->execute_index_space(ctx, launchsfq);

    double svdargs[] = { qcs->q1, qcs->q2 };
    IndexLauncher launchsvd(TID_SETVELDIFF, dompc,
            TaskArgument(svdargs, sizeof(svdargs)), am);
    launchsvd.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchsvd.add_field(0, FID_MAPSZ);
    launchsvd.add_field(0, FID_MAPSP1);
    launchsvd.add_field(0, FID_MAPSP2);
    launchsvd.add_field(0, FID_MAPSP1REG);
    launchsvd.add_field(0, FID_MAPSP2REG);
    launchsvd.add_field(0, FID_ELEN);
    launchsvd.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchsvd.add_field(1, FID_ZSS);
    launchsvd.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchsvd.add_field(2, FID_PXP);
    launchsvd.add_field(2, FID_PU0);
    launchsvd.add_region_requirement(
            RegionRequirement(lppshr, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchsvd.add_field(3, FID_PXP);
    launchsvd.add_field(3, FID_PU0);
    launchsvd.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchsvd.add_field(4, FID_ZTMP);
    launchsvd.add_field(4, FID_ZDU);
    // requires read only access to shared (ghost) and private point data 
    runtime->execute_index_space(ctx, launchsvd);

    // this sums corner forces to points 
    IndexLauncher launchscf(TID_SUMCRNRFORCE, dompc, ta, am);
    launchscf.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchscf.add_field(0, FID_MAPSP1);
    launchscf.add_field(0, FID_MAPSP1REG);
    launchscf.add_field(0, FID_MAPSS3);
    launchscf.add_field(0, FID_SFP);
    launchscf.add_field(0, FID_SFQ);
    launchscf.add_field(0, FID_SFT);
    launchscf.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_WRITE, EXCLUSIVE, lrp));
    launchscf.add_field(1, FID_PF);
    launchscf.add_region_requirement(
            RegionRequirement(lppshr, 0, OPID_SUMDBL2,
                    SIMULTANEOUS, lrp));
    launchscf.add_field(2, FID_PF);
    // requires read / write access to private and shared (ghosted) points
    //   uses a reduction operation to sum corner masses to point mass 
    runtime->execute_index_space(ctx, launchscf);

    // 4a. apply boundary conditions
    IndexLauncher launchafbc(TID_APPLYFIXEDBC, dompc, ta, am);
    for (int i = 0; i < bcs.size(); ++i) {
      // 
        double2 afbcargs[1] = { bcs[i]->vfix };
        launchafbc.global_arg =
                TaskArgument(afbcargs, sizeof(afbcargs));
        launchafbc.region_requirements.clear();
        launchafbc.add_region_requirement(
                RegionRequirement(bcs[i]->lpb, 0,
                        READ_ONLY, EXCLUSIVE, bcs[i]->lrb));
        launchafbc.add_field(0, FID_MAPBP);
        launchafbc.add_field(0, FID_MAPBPREG);
        launchafbc.add_region_requirement(
                RegionRequirement(lppprv, 0,
                        READ_WRITE, EXCLUSIVE, lrp));
        launchafbc.add_field(1, FID_PF);
        launchafbc.add_field(1, FID_PU0);
        launchafbc.add_region_requirement(
                RegionRequirement(lppmstr, 0,
                        READ_WRITE, EXCLUSIVE, lrp));
        launchafbc.add_field(2, FID_PF);
        launchafbc.add_field(2, FID_PU0);
        // this task updates point values both private and ghosted (using master partition) 
        runtime->execute_index_space(ctx, launchafbc);
    }

    // check for negative volumes on predictor step
    mesh->checkBadSides();

    IndexLauncher launchca(TID_CALCACCEL, dompc, ta, am);
    double apfargs[] = { dt };
    IndexLauncher launchapf(TID_ADVPOSFULL, dompc,
            TaskArgument(apfargs, sizeof(apfargs)), am);
    // do point routines twice, once each for private and master
    // partitions
    for (int part = 0; part < 2; ++part) {
        LogicalPartition& lppcurr = (part == 0 ? lppprv : lppmstr);

        // 5. compute accelerations
        launchca.region_requirements.clear();
        launchca.add_region_requirement(
                RegionRequirement(lppcurr, 0,
                        READ_ONLY, EXCLUSIVE, lrp));
        launchca.add_field(0, FID_PF);
        launchca.add_field(0, FID_PMASWT);
        launchca.add_region_requirement(
                RegionRequirement(lppcurr, 0,
                        WRITE_DISCARD, EXCLUSIVE, lrp));
        launchca.add_field(1, FID_PAP);
        // this updates point acceleration values on private and master
        //   partitions 
        runtime->execute_index_space(ctx, launchca);

        // ===== Corrector step =====
        // 6. advance mesh to end of time step
        launchapf.region_requirements.clear();
        launchapf.add_region_requirement(
                RegionRequirement(lppcurr, 0,
                        READ_ONLY, EXCLUSIVE, lrp));
        launchapf.add_field(0, FID_PX0);
        launchapf.add_field(0, FID_PU0);
        launchapf.add_field(0, FID_PAP);
        launchapf.add_region_requirement(
                RegionRequirement(lppcurr, 0,
                        WRITE_DISCARD, EXCLUSIVE, lrp));
        launchapf.add_field(1, FID_PX);
        launchapf.add_field(1, FID_PU);
        // this updates point coordintate and velocity on private
        //   and master partitions 
        runtime->execute_index_space(ctx, launchapf);
    }  // for part

    // 6a. compute new mesh geometry
    // reuse launchers from earlier, with corrector-step fields
    for (int r = 2; r < 6; ++r) {
        launchcc.region_requirements[r].privilege_fields.clear();
        launchcc.region_requirements[r].instance_fields.clear();
    }
    launchcc.add_field(2, FID_PX);
    launchcc.add_field(3, FID_PX);
    launchcc.add_field(4, FID_EX);
    launchcc.add_field(5, FID_ZX);
    // requires read access to ghost points, updates private zones and sides 
    runtime->execute_index_space(ctx, launchcc);

    for (int r = 1; r < 6; ++r) {
        launchcv.region_requirements[r].privilege_fields.clear();
        launchcv.region_requirements[r].instance_fields.clear();
    }
    launchcv.add_field(1, FID_PX);
    launchcv.add_field(2, FID_PX);
    launchcv.add_field(3, FID_ZX);
    launchcv.add_field(4, FID_SAREA);
    launchcv.add_field(4, FID_SVOL);
    launchcv.add_field(5, FID_ZAREA);
    launchcv.add_field(5, FID_ZVOL);
    // requires read access to ghost points, updates private zones and sides 
    mesh->fmapcv = runtime->execute_index_space(ctx, launchcv);

    // 7. compute work
    double cwargs[] = { dt };
    IndexLauncher launchcw(TID_CALCWORK, dompc,
            TaskArgument(cwargs, sizeof(cwargs)), am);
    launchcw.add_region_requirement(
            RegionRequirement(lps, 0, READ_ONLY, EXCLUSIVE, lrs));
    launchcw.add_field(0, FID_MAPSP1);
    launchcw.add_field(0, FID_MAPSP2);
    launchcw.add_field(0, FID_MAPSZ);
    launchcw.add_field(0, FID_MAPSP1REG);
    launchcw.add_field(0, FID_MAPSP2REG);
    launchcw.add_field(0, FID_SFP);
    launchcw.add_field(0, FID_SFQ);
    launchcw.add_region_requirement(
            RegionRequirement(lppprv, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchcw.add_field(1, FID_PU);
    launchcw.add_field(1, FID_PU0);
    launchcw.add_field(1, FID_PXP);
    launchcw.add_region_requirement(
            RegionRequirement(lppshr, 0, READ_ONLY, EXCLUSIVE, lrp));
    launchcw.add_field(2, FID_PU);
    launchcw.add_field(2, FID_PU0);
    launchcw.add_field(2, FID_PXP);
    launchcw.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchcw.add_field(3, FID_ZW);
    launchcw.add_region_requirement(
            RegionRequirement(lpz, 0, READ_WRITE, EXCLUSIVE, lrz));
    launchcw.add_field(4, FID_ZETOT);
    // requires read access to ghost points, updates private zones and sides 
    runtime->execute_index_space(ctx, launchcw);

    double cwrargs[] = { dt };
    IndexLauncher launchcwr(TID_CALCWORKRATE, dompc,
            TaskArgument(cwrargs, sizeof(cwrargs)), am);
    launchcwr.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcwr.add_field(0, FID_ZVOL0);
    launchcwr.add_field(0, FID_ZVOL);
    launchcwr.add_field(0, FID_ZW);
    launchcwr.add_field(0, FID_ZP);
    launchcwr.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchcwr.add_field(1, FID_ZWRATE);
    // no ghost access required 
    runtime->execute_index_space(ctx, launchcwr);

    // 8. update state variables
    IndexLauncher launchce(TID_CALCENERGY, dompc, ta, am);
    launchce.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchce.add_field(0, FID_ZETOT);
    launchce.add_field(0, FID_ZM);
    launchce.add_region_requirement(
            RegionRequirement(lpz, 0, WRITE_DISCARD, EXCLUSIVE, lrz));
    launchce.add_field(1, FID_ZE);
    // no ghost access required 
    runtime->execute_index_space(ctx, launchce);

    // reuse launcher from earlier, with corrector-step fields
    for (int r = 0; r < 2; ++r) {
        launchcr.region_requirements[r].privilege_fields.clear();
        launchcr.region_requirements[r].instance_fields.clear();
    }
    launchcr.add_field(0, FID_ZM);
    launchcr.add_field(0, FID_ZVOL);
    launchcr.add_field(1, FID_ZR);
    // no ghost access required 
    runtime->execute_index_space(ctx, launchcr);

    // 9.  compute timestep for next cycle
    double cdtargs[] = { cfl, cflv, dt };
    IndexLauncher launchcdt(TID_CALCDT, dompc,
            TaskArgument(cdtargs, sizeof(cdtargs)), am);
    launchcdt.add_region_requirement(
            RegionRequirement(lpz, 0, READ_ONLY, EXCLUSIVE, lrz));
    launchcdt.add_field(0, FID_ZDL);
    launchcdt.add_field(0, FID_ZDU);
    launchcdt.add_field(0, FID_ZSS);
    launchcdt.add_field(0, FID_ZVOL);
    launchcdt.add_field(0, FID_ZVOL0);
    // no ghost access required 
    fmapcdt = runtime->execute_index_space(ctx, launchcdt);

    runtime->end_trace(ctx, 123);

    // check for negative volumes on corrector step
    // this checks the return values from the CALCVOLSTASK
    mesh->checkBadSides();
}


void Hydro::getFinalState() {
    mesh->getField(mesh->lrp, FID_PX, mesh->px, mesh->nump);
    mesh->getField(mesh->lrz, FID_ZP, zp, mesh->numz);
    mesh->getField(mesh->lrz, FID_ZE, ze, mesh->numz);
    mesh->getField(mesh->lrz, FID_ZR, zr, mesh->numz);
}



void Hydro::advPosHalfTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
    const double dt = *((const double*)task->args);
    const double dth = 0.5 * dt;
    MyAccessor<double2> acc_px0 =
        get_accessor<double2>(regions[0], FID_PX0);
    MyAccessor<double2> acc_pu0 =
        get_accessor<double2>(regions[0], FID_PU0);
    MyAccessor<double2> acc_pxp =
        get_accessor<double2>(regions[1], FID_PXP);

    const IndexSpace& isp = task->regions[0].region.get_index_space();
        
    for (IndexIterator itrp(runtime, ctx, isp); itrp.has_next(); )
    {
        ptr_t p = itrp.next();
        double2 x0 = acc_px0.read(p);
        double2 u0 = acc_pu0.read(p);
        double2 xp = x0 + dth * u0;
        acc_pxp.write(p, xp);
    }
}


void Hydro::calcRhoTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
    FieldID fid_zm = task->regions[0].instance_fields[0];
    FieldID fid_zvol = task->regions[0].instance_fields[1];
    MyAccessor<double> acc_zm =
        get_accessor<double>(regions[0], fid_zm);
    MyAccessor<double> acc_zvol =
        get_accessor<double>(regions[0], fid_zvol);
    FieldID fid_zr = task->regions[1].instance_fields[0];
    MyAccessor<double> acc_zr =
        get_accessor<double>(regions[1], fid_zr);

    const IndexSpace& isz = task->regions[0].region.get_index_space();
    for (IndexIterator itrz(runtime, ctx, isz); itrz.has_next();)
    {
        ptr_t z = itrz.next();
        double m = acc_zm.read(z);
        double v = acc_zvol.read(z);
        double r = m / v;
        acc_zr.write(z, r);
    }

}


void Hydro::calcCrnrMassTask( // This updates ghosts (2. compute point masses) 
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
    MyAccessor<ptr_t> acc_mapsp1 =
        get_accessor<ptr_t>(regions[0], FID_MAPSP1);
    MyAccessor<int> acc_mapsp1reg =
        get_accessor<int>(regions[0], FID_MAPSP1REG);
    MyAccessor<ptr_t> acc_mapss3 =
        get_accessor<ptr_t>(regions[0], FID_MAPSS3);
    MyAccessor<ptr_t> acc_mapsz =
        get_accessor<ptr_t>(regions[0], FID_MAPSZ);
    MyAccessor<double> acc_smf =
        get_accessor<double>(regions[0], FID_SMF);
    MyAccessor<double> acc_zr =
        get_accessor<double>(regions[1], FID_ZRP);
    MyAccessor<double> acc_zarea =
        get_accessor<double>(regions[1], FID_ZAREAP);
    MyAccessor<double> acc_pmas_prv =
        get_accessor<double>(regions[2], FID_PMASWT);
    // the next region has simultaneous access, using a
    //   reduction to sum corner masses to point mass 
    MyReductionAccessor<SumOp<double> > acc_pmas_shr =
      get_reduction_accessor<SumOp<double> >(regions[3]);

    const IndexSpace& iss = task->regions[0].region.get_index_space();

    for (IndexIterator itrs(runtime, ctx, iss); itrs.has_next();)
    {
        ptr_t s  = itrs.next();
        ptr_t s3 = acc_mapss3.read(s);
        ptr_t z  = acc_mapsz.read(s);
        ptr_t p = acc_mapsp1.read(s);
        int preg = acc_mapsp1reg.read(s);
        double r = acc_zr.read(z);
        double area = acc_zarea.read(z);
        double mf = acc_smf.read(s);
        double mf3 = acc_smf.read(s3);
        double mwt = r * area * 0.5 * (mf + mf3);
        if (preg == 0)
            acc_pmas_prv.reduce<SumOp<double> >(p, mwt);
        else
            acc_pmas_shr.reduce(p, mwt);
    }
}


void Hydro::sumCrnrForceTask( // this updates ghosts (4. compute point forces) 
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
    MyAccessor<ptr_t> acc_mapsp1 =
        get_accessor<ptr_t>(regions[0], FID_MAPSP1);
    MyAccessor<int> acc_mapsp1reg =
        get_accessor<int>(regions[0], FID_MAPSP1REG);
    MyAccessor<ptr_t> acc_mapss3 =
        get_accessor<ptr_t>(regions[0], FID_MAPSS3);
    MyAccessor<double2> acc_sfp =
        get_accessor<double2>(regions[0], FID_SFP);
    MyAccessor<double2> acc_sfq =
        get_accessor<double2>(regions[0], FID_SFQ);
    MyAccessor<double2> acc_sft =
        get_accessor<double2>(regions[0], FID_SFT);
    MyAccessor<double2> acc_pf_prv =
        get_accessor<double2>(regions[1], FID_PF);
    MyReductionAccessor<SumOp<double2> > acc_pf_shr =
        get_reduction_accessor<SumOp<double2> >(regions[2]);

    const IndexSpace& iss = task->regions[0].region.get_index_space();

    for (IndexIterator itrs(runtime, ctx, iss); itrs.has_next(); )
    {
        ptr_t s  = itrs.next();
        ptr_t s3 = acc_mapss3.read(s);
        ptr_t p = acc_mapsp1.read(s);
        int preg = acc_mapsp1reg.read(s);
        double2 sfp = acc_sfp.read(s);
        double2 sfq = acc_sfq.read(s);
        double2 sft = acc_sft.read(s);
        double2 sfp3 = acc_sfp.read(s3);
        double2 sfq3 = acc_sfq.read(s3);
        double2 sft3 = acc_sft.read(s3);
        double2 cf = (sfp + sfq + sft) - (sfp3 + sfq3 + sft3);
        if (preg == 0)
            acc_pf_prv.reduce<SumOp<double2> >(p, cf);
        else
            acc_pf_shr.reduce(p, cf);
    }
}


void Hydro::calcAccelTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
    MyAccessor<double2> acc_pf =
        get_accessor<double2>(regions[0], FID_PF);
    MyAccessor<double> acc_pmass =
        get_accessor<double>(regions[0], FID_PMASWT);
    MyAccessor<double2> acc_pa =
        get_accessor<double2>(regions[1], FID_PAP);

    const double fuzz = 1.e-99;
    const IndexSpace& isp = task->regions[0].region.get_index_space();

    for (IndexIterator itrp(runtime, ctx, isp); itrp.has_next(); )
    {
        ptr_t p = itrp.next();
        double2 f = acc_pf.read(p);
        double m = acc_pmass.read(p);
        double2 a = f / max(m, fuzz);
        acc_pa.write(p, a);
    }
}


void Hydro::advPosFullTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
    const double dt = *((const double*)task->args);

    MyAccessor<double2> acc_px0 =
        get_accessor<double2>(regions[0], FID_PX0);
    MyAccessor<double2> acc_pu0 =
        get_accessor<double2>(regions[0], FID_PU0);
    MyAccessor<double2> acc_pa =
        get_accessor<double2>(regions[0], FID_PAP);
    MyAccessor<double2> acc_px =
        get_accessor<double2>(regions[1], FID_PX);
    MyAccessor<double2> acc_pu =
        get_accessor<double2>(regions[1], FID_PU);

    const IndexSpace& isp = task->regions[0].region.get_index_space();

    for (IndexIterator itrp(runtime, ctx, isp); itrp.has_next(); )
    {
        ptr_t p = itrp.next();
        double2 x0 = acc_px0.read(p);
        double2 u0 = acc_pu0.read(p);
        double2 a = acc_pa.read(p);
        double2 u = u0 + dt * a;
        acc_pu.write(p, u);
        double2 x = x0 + dt * 0.5 * (u0 + u);
        acc_px.write(p, x);
    }

}


void Hydro::calcWorkTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
    const double dt = *((const double*)task->args);

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
    MyAccessor<double2> acc_sf =
        get_accessor<double2>(regions[0], FID_SFP);
    MyAccessor<double2> acc_sf2 =
        get_accessor<double2>(regions[0], FID_SFQ);
    MyAccessor<double2> acc_pu0[2] = {
        get_accessor<double2>(regions[1], FID_PU0),
        get_accessor<double2>(regions[2], FID_PU0)
    };
    MyAccessor<double2> acc_pu[2] = {
        get_accessor<double2>(regions[1], FID_PU),
        get_accessor<double2>(regions[2], FID_PU)
    };
    MyAccessor<double2> acc_px[2] = {
        get_accessor<double2>(regions[1], FID_PXP),
        get_accessor<double2>(regions[2], FID_PXP)
    };
    MyAccessor<double> acc_zw =
        get_accessor<double>(regions[3], FID_ZW);
    MyAccessor<double> acc_zetot =
        get_accessor<double>(regions[4], FID_ZETOT);

    // Compute the work done by finding, for each element/node pair,
    //   dwork= force * vavg
    // where force is the force of the element on the node
    // and vavg is the average velocity of the node over the time period

    const IndexSpace& isz = task->regions[3].region.get_index_space();

    for (IndexIterator itrz(runtime, ctx, isz); itrz.has_next(); )
    {
        ptr_t z = itrz.next();
        acc_zw.write(z, 0.);

    }

    const double dth = 0.5 * dt;

    const IndexSpace& iss = task->regions[0].region.get_index_space();
 
    for (IndexIterator itrs(runtime, ctx, iss); itrs.has_next();)
    {
        ptr_t s = itrs.next();
        ptr_t p1 = acc_mapsp1.read(s);
        int p1reg = acc_mapsp1reg.read(s);
        ptr_t p2 = acc_mapsp2.read(s);
        int p2reg = acc_mapsp2reg.read(s);
        ptr_t z  = acc_mapsz.read(s);
        double2 sf = acc_sf.read(s);
        double2 sf2 = acc_sf2.read(s);
        double2 sftot = sf + sf2;
        double2 pu01 = acc_pu0[p1reg].read(p1);
        double2 pu1 = acc_pu[p1reg].read(p1);
        double sd1 = dot(sftot, (pu01 + pu1));
        double2 pu02 = acc_pu0[p2reg].read(p2);
        double2 pu2 = acc_pu[p2reg].read(p2);
        double sd2 = dot(-sftot, (pu02 + pu2));
        double2 px1 = acc_px[p1reg].read(p1);
        double2 px2 = acc_px[p2reg].read(p2);
        double dwork = -dth * (sd1 * px1.x + sd2 * px2.x);

        double zetot = acc_zetot.read(z);
        zetot += dwork;
        acc_zetot.write(z, zetot);
        double zw = acc_zw.read(z);
        zw += dwork;
        acc_zw.write(z, zw);
    }

}


void Hydro::calcWorkRateTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
    const double dt = *((const double*)task->args);

    MyAccessor<double> acc_zvol0 =
        get_accessor<double>(regions[0], FID_ZVOL0);
    MyAccessor<double> acc_zvol =
        get_accessor<double>(regions[0], FID_ZVOL);
    MyAccessor<double> acc_zw =
        get_accessor<double>(regions[0], FID_ZW);
    MyAccessor<double> acc_zp =
        get_accessor<double>(regions[0], FID_ZP);
    MyAccessor<double> acc_zwrate =
        get_accessor<double>(regions[1], FID_ZWRATE);

    double dtinv = 1. / dt;

    const IndexSpace& isz = task->regions[0].region.get_index_space();

    for (IndexIterator itrz(runtime, ctx, isz); itrz.has_next();)
    {
        ptr_t z = itrz.next();
        double zvol = acc_zvol.read(z);
        double zvol0 = acc_zvol0.read(z);
        double dvol = zvol - zvol0;
        double zw = acc_zw.read(z);
        double zp = acc_zp.read(z);
        double zwrate = (zw + zp * dvol) * dtinv;
        acc_zwrate.write(z, zwrate);
    }
}


void Hydro::calcEnergyTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
    MyAccessor<double> acc_zetot =
        get_accessor<double>(regions[0], FID_ZETOT);
    MyAccessor<double> acc_zm =
        get_accessor<double>(regions[0], FID_ZM);
    MyAccessor<double> acc_ze =
        get_accessor<double>(regions[1], FID_ZE);

    const double fuzz = 1.e-99;
    const IndexSpace& isz = task->regions[0].region.get_index_space();

    for (IndexIterator itrz(runtime, ctx, isz); itrz.has_next(); )
    {
        ptr_t z = itrz.next();
        double zetot = acc_zetot.read(z);
        double zm = acc_zm.read(z);
        double ze = zetot / (zm + fuzz);
        acc_ze.write(z, ze);
    }

}


double Hydro::calcDtTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        HighLevelRuntime *runtime) {
    const double* args = (const double*) task->args;
    const double cfl    = args[0];
    const double cflv   = args[1];
    const double dtlast = args[2];

    MyAccessor<double> acc_zdl =
        get_accessor<double>(regions[0], FID_ZDL);
    MyAccessor<double> acc_zdu =
        get_accessor<double>(regions[0], FID_ZDU);
    MyAccessor<double> acc_zss =
        get_accessor<double>(regions[0], FID_ZSS);
    MyAccessor<double> acc_zvol =
        get_accessor<double>(regions[0], FID_ZVOL);
    MyAccessor<double> acc_zvol0 =
        get_accessor<double>(regions[0], FID_ZVOL0);

    double dtrec = 1.e99;

    // compute dt using Courant condition
    const double fuzz = 1.e-99;
    double dtnew = 1.e99;
    int zmin = -1;
    const IndexSpace& isz = task->regions[0].region.get_index_space();
    

    for (IndexIterator itrz(runtime, ctx, isz); itrz.has_next(); )
    {
        ptr_t z = itrz.next();
        double zdu = acc_zdu.read(z);
        double zss = acc_zss.read(z);
        double cdu = max(zdu, max(zss, fuzz));
        double zdl = acc_zdl.read(z);
        double zdthyd = zdl * cfl / cdu;
        zmin = (zdthyd < dtnew ? (int) z : zmin);
        dtnew = (zdthyd < dtnew ? zdthyd : dtnew);
    }
    if (dtnew < dtrec) {
        dtrec = dtnew;
    }

    // compute dt using volume condition
    double dvovmax = 1.e-99;
    int zmax = -1;
    

    for (IndexIterator itrz(runtime, ctx, isz); itrz.has_next();)
    {
        ptr_t z = itrz.next();
        double zvol = acc_zvol.read(z);
        double zvol0 = acc_zvol0.read(z);
        double zdvov = abs((zvol - zvol0) / zvol0);
        zmax = (zdvov > dvovmax ? (int) z : zmax);
        dvovmax = (zdvov > dvovmax ? zdvov : dvovmax);
    }
    double dtnew2 = dtlast * cflv / dvovmax;
    if (dtnew2 < dtrec) {
        dtrec = dtnew2;
    }

    return dtrec;
}


void Hydro::getDtHydro(
        double& dtnew,
        string& msgdtnew) {

    dtrec = mesh->reduceFutureMap<MinOp<double> >(fmapcdt);
    if (dtrec < dtnew) {
        dtnew = dtrec;
        msgdtnew = "Hydro timestep";
    }

}


void Hydro::resetDtHydro() {

//    dtrec = 1.e99;
//    msgdtrec = "Hydro default";

}
